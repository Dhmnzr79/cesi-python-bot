"""Chunk selection orchestration for /ask retrieval path."""
import json
import os
import re

from config import (
    ALIAS_SOFT_THRESHOLD,
    ALIAS_STRONG_THRESHOLD,
    LOW_SCORE_THRESHOLD,
    QUERY_REWRITE_ON,
    DEFAULT_CLIENT_ID,
    PRICE_CONCERN_RE,
    PRICE_LOOKUP_RE,
    PRICE_SERVICE_MATCH_STRONG,
    RERANK_GAP_MAX,
    RERANK_NEAR_LOW_GAP_MAX,
    RERANK_NEAR_LOW_TOP_MAX,
    RERANK_TOP_MAX,
    RERANK_TOP_MIN,
)
import alias_lexical
from llm import classify_price_intent, rewrite_query_for_retrieval
from session import mem_get
from policy import contacts_intent, pick_contacts_chunk, pick_prices_chunk, price_intent
from retriever import (
    broad_query_detect,
    corpus_alias_leader,
    is_point_literal_query,
    llm_rerank,
    merge_retrieval_candidates,
    normalize_retrieval_query,
    prefer_overview_if_broad,
    retrieve,
)


def select_chunk_for_question(
    q: str, *, client_id: str | None, sid: str | None = None
) -> dict:
    """Return selection result for /ask.

    mode:
      - no_candidates
      - low_score
      - chunk
    """
    q_user = (q or "").strip()
    if sid and QUERY_REWRITE_ON:
        q_rewrite_eff = rewrite_query_for_retrieval(sid, q_user, client_id=client_id)
    else:
        q_rewrite_eff = q_user

    # Интенты и алиасы — только по исходному вопросу пациента (не по rewrite).
    q_policy = normalize_retrieval_query(q_user) or q_user
    nu = (normalize_retrieval_query(q_user) or q_user).strip().lower()
    nr = (normalize_retrieval_query(q_rewrite_eff) or q_rewrite_eff).strip().lower()

    nr_meta = normalize_retrieval_query(q_rewrite_eff) or q_rewrite_eff
    base_meta = {
        "query_user_raw": q_user[:200],
        "query_rewrite_effective": q_rewrite_eff[:200],
        "query_normalized_user": q_policy[:200],
        "query_normalized_rewrite": nr_meta[:200],
        "rewrite_applied": bool(q_rewrite_eff.strip().lower() != q_user.strip().lower()),
    }

    def _dm(extra: dict) -> dict:
        return {**base_meta, **extra}

    primary = retrieve(q_user, topk=8, client_id=client_id)
    secondary: list = []
    if nr != nu:
        secondary = retrieve(
            q_rewrite_eff, topk=8, client_id=client_id, silent=True
        )
    cands = merge_retrieval_candidates(primary, secondary)[:8]
    cands = prefer_overview_if_broad(cands, broad_query_detect(q_policy))
    if not cands:
        return {
            "mode": "no_candidates",
            "debug_meta": _dm({"top_score": None}),
        }

    is_contacts = contacts_intent(q_policy)
    is_price = price_intent(q_policy)
    alias_leader, alias_score = corpus_alias_leader(q_policy, client_id=client_id)
    alias_strong = bool(alias_leader and alias_score >= ALIAS_STRONG_THRESHOLD)

    top_score = float(cands[0].get("_score") or 0.0)
    allow_low = alias_strong or (is_contacts and pick_contacts_chunk(cands)) or (
        is_price and pick_prices_chunk(cands)
    )
    if top_score < LOW_SCORE_THRESHOLD and not allow_low:
        if alias_leader and alias_score >= ALIAS_SOFT_THRESHOLD:
            soft = dict(alias_leader)
            soft["_alias_score"] = round(alias_score, 4)
            soft["_score"] = round(float(alias_score), 4)
            return {
                "mode": "chunk",
                "chunk": soft,
                "rerank_applied": False,
                "debug_meta": _dm(
                    {
                        "selected_by": "soft_alias_assist",
                        "top_score": round(top_score, 4),
                        "threshold": LOW_SCORE_THRESHOLD,
                        "alias_score": round(float(alias_score or 0.0), 4),
                        "is_contacts": bool(is_contacts),
                        "is_price": bool(is_price),
                    }
                ),
            }
        return {
            "mode": "low_score",
            "debug_meta": _dm(
                {
                    "top_score": round(top_score, 4),
                    "threshold": LOW_SCORE_THRESHOLD,
                    "alias_score": round(float(alias_score or 0.0), 4),
                    "is_contacts": bool(is_contacts),
                    "is_price": bool(is_price),
                }
            ),
        }

    if is_contacts:
        picked = pick_contacts_chunk(cands)
        if picked is not None:
            return {
                "mode": "chunk",
                "chunk": picked,
                "rerank_applied": False,
                "debug_meta": _dm(
                    {
                        "selected_by": "contacts",
                        "top_score": round(top_score, 4),
                        "alias_score": round(float(alias_score or 0.0), 4),
                    }
                ),
            }

    if is_price:
        picked = pick_prices_chunk(cands)
        if picked is not None:
            return {
                "mode": "chunk",
                "chunk": picked,
                "rerank_applied": False,
                "debug_meta": _dm(
                    {
                        "selected_by": "price",
                        "top_score": round(top_score, 4),
                        "alias_score": round(float(alias_score or 0.0), 4),
                    }
                ),
            }

    if alias_strong and alias_leader is not None:
        strong = dict(alias_leader)
        strong["_alias_score"] = round(alias_score, 4)
        strong["_score"] = round(float(alias_score), 4)
        return {
            "mode": "chunk",
            "chunk": strong,
            "rerank_applied": False,
            "debug_meta": _dm(
                {
                    "selected_by": "alias",
                    "alias_score": round(float(alias_score or 0.0), 4),
                    "top_score": round(top_score, 4),
                }
            ),
        }

    top = cands[0]
    score_gap = (
        abs(float(cands[0].get("_score") or 0.0) - float(cands[1].get("_score") or 0.0))
        if len(cands) >= 2
        else 1.0
    )
    narrow_rerank = (
        RERANK_TOP_MIN <= top_score <= RERANK_TOP_MAX
        and len(cands) >= 2
        and score_gap <= RERANK_GAP_MAX
        and not is_point_literal_query(q_policy)
        and not alias_strong
    )
    near_low_rerank = (
        len(cands) >= 2
        and RERANK_TOP_MIN <= top_score <= RERANK_NEAR_LOW_TOP_MAX
        and top_score < LOW_SCORE_THRESHOLD + 0.02
        and score_gap <= RERANK_NEAR_LOW_GAP_MAX
        and not is_point_literal_query(q_policy)
        and not alias_strong
    )
    use_rerank = narrow_rerank or near_low_rerank
    if use_rerank:
        top = llm_rerank(q_user, cands[:3])

    return {
        "mode": "chunk",
        "chunk": top,
        "rerank_applied": bool(use_rerank),
        "debug_meta": _dm(
            {
                "selected_by": "semantic",
                "top_score": round(top_score, 4),
                "score_gap": round(float(score_gap), 4),
                "alias_score": round(float(alias_score or 0.0), 4),
                "rerank_near_low": bool(near_low_rerank and not narrow_rerank),
            }
        ),
    }


def _safe_client_id(client_id: str | None) -> str:
    return (client_id or DEFAULT_CLIENT_ID or "default").strip() or "default"


def _client_json_path(client_id: str | None, file_name: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "clients", _safe_client_id(client_id), file_name)


def _read_json_dict(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except OSError:
        return {}
    except json.JSONDecodeError:
        return {}


def _norm(s: str) -> str:
    x = (s or "").strip().lower().replace("ё", "е")
    x = re.sub(r"[^\w\s]", " ", x, flags=re.U)
    return re.sub(r"\s+", " ", x, flags=re.U).strip()


_STOP = frozenset({
    "а", "в", "во", "на", "по", "за", "к", "ко", "с", "со", "из", "от", "до",
    "не", "ли", "бы", "же", "и", "или", "но", "что", "как", "это", "для",
    "при", "под", "над", "без", "то", "все", "мне", "мой", "моя", "моё",
    "вы", "вас", "вам", "нас", "нам", "их", "его", "её",
})


def _token_set(s: str) -> set[str]:
    return {t for t in _norm(s).split() if len(t) >= 2 or t.isdigit()}


def _core_tokens_catalog(text: str) -> list[str]:
    return [t for t in _norm(text).split() if (len(t) >= 2 or t.isdigit()) and t not in _STOP]


def _match_score_lemma(query: str, phrase: str) -> float:
    """Матч с лемматизацией — обрабатывает падежи ("коронки" = "коронка")."""
    q_toks = _core_tokens_catalog(query)
    p_toks = _core_tokens_catalog(phrase)
    if not q_toks or not p_toks:
        return 0.0
    q_lem = set(alias_lexical.lemma_forms_for_tokens(q_toks))
    p_lem = set(alias_lexical.lemma_forms_for_tokens(p_toks))
    if not q_lem or not p_lem:
        return 0.0
    # Все леммы alias-фразы входят в запрос — сильный матч
    if p_lem <= q_lem:
        return 0.92
    # Все леммы запроса входят в alias-фразу
    if q_lem <= p_lem:
        return 0.88
    inter = len(q_lem & p_lem)
    if inter == 0:
        return 0.0
    recall = inter / len(p_lem)
    precision = inter / len(q_lem)
    return round(max(recall, (recall + precision) / 2.0), 4)


def _match_score(query: str, phrase: str) -> float:
    qn = _norm(query)
    pn = _norm(phrase)
    if not qn or not pn:
        return 0.0
    if pn in qn:
        return 1.0
    qt = _token_set(qn)
    pt = _token_set(pn)
    if not qt or not pt:
        return 0.0
    inter = len(qt.intersection(pt))
    if inter == 0:
        return 0.0
    recall = inter / len(pt)
    precision = inter / len(qt)
    return round(max(recall, (recall + precision) / 2.0), 4)


def _lookup_intent_by_rules(q: str) -> str:
    q0 = (q or "").strip()
    if not q0:
        return "other"
    if PRICE_CONCERN_RE.search(q0):
        return "price_concern"
    if PRICE_LOOKUP_RE.search(q0):
        return "price_lookup"
    return "other"


def classify_price_route_intent(q: str, *, client_id: str | None, sid: str | None) -> str:
    rule_intent = _lookup_intent_by_rules(q)
    if rule_intent != "other":
        return rule_intent
    return classify_price_intent(q, client_id=client_id, sid=sid or "")


def match_service_from_catalog(q: str, *, client_id: str | None) -> dict:
    catalog = _read_json_dict(_client_json_path(client_id, "service_catalog.json"))
    best_id = None
    best_obj = None
    best_score = 0.0
    for service_id, entry in catalog.items():
        if not isinstance(entry, dict) or not bool(entry.get("active", True)):
            continue
        phrases = []
        title = str(entry.get("title") or "").strip()
        if title:
            phrases.append(title)
        aliases = list(entry.get("aliases") or [])
        phrases.extend(str(x).strip() for x in aliases if str(x).strip())
        local_best = 0.0
        for ph in phrases:
            local_best = max(local_best, _match_score(q, ph), _match_score_lemma(q, ph))
        if local_best > best_score:
            best_id = str(service_id)
            best_obj = entry
            best_score = local_best
    return {
        "matched_service_id": best_id,
        "service": best_obj,
        "match_score": round(float(best_score), 4),
        "is_confident": bool(best_obj is not None and best_score >= PRICE_SERVICE_MATCH_STRONG),
    }


def _service_from_session_context(sid: str | None, client_id: str | None) -> dict | None:
    """Ищет услугу в каталоге по current_doc_id или last_catalog_service_id из сессии.

    Возвращает dict {service_id, service, price_key, price_ref, price_item} или None.
    Используется как fallback когда пользователь спрашивает цену без названия услуги,
    но до этого уже смотрел конкретную услугу.
    """
    if not sid:
        return None
    st = mem_get(sid)
    catalog = _read_json_dict(_client_json_path(client_id, "service_catalog.json"))
    if not isinstance(catalog, dict):
        return None

    def _make_result(service_id: str, entry: dict, context_doc_id: str | None) -> dict:
        prices = _read_json_dict(_client_json_path(client_id, "prices.json"))
        price_key = entry.get("price_key")
        price_ref = entry.get("price_ref")
        price_item = prices.get(price_key) if isinstance(prices, dict) and price_key else None
        return {
            "service_id": str(service_id),
            "service": entry,
            "price_key": price_key,
            "price_ref": price_ref,
            "price_item": price_item if isinstance(price_item, dict) else None,
            "context_doc_id": context_doc_id,
        }

    # Попытка 1: по current_doc_id (сервисы с md_entry_ref)
    current_doc_id = (st.get("current_doc_id") or "").strip()
    if current_doc_id:
        doc_norm = current_doc_id.removesuffix(".md")
        for service_id, entry in catalog.items():
            if not isinstance(entry, dict) or not bool(entry.get("active", True)):
                continue
            md_ref = (entry.get("md_entry_ref") or "").strip()
            if not md_ref:
                continue
            if md_ref.removesuffix(".md") == doc_norm:
                return _make_result(service_id, entry, current_doc_id)

    # Попытка 2: по last_catalog_service_id (сервисы без md_entry_ref, напр. КТ, отбеливание)
    last_svc_id = (st.get("last_catalog_service_id") or "").strip()
    if last_svc_id and last_svc_id in catalog:
        entry = catalog[last_svc_id]
        if isinstance(entry, dict) and bool(entry.get("active", True)):
            return _make_result(last_svc_id, entry, None)

    return None


def select_price_service_route(
    q: str, *, client_id: str | None, sid: str | None = None
) -> dict:
    intent = classify_price_route_intent(q, client_id=client_id, sid=sid)
    if intent == "other":
        return {"mode": "other", "intent": intent}
    match = match_service_from_catalog(q, client_id=client_id)
    if not match.get("matched_service_id"):
        ctx = _service_from_session_context(sid, client_id)
        if ctx and intent == "price_lookup":
            return {
                "mode": "matched",
                "intent": intent,
                "route_source": "prices_json" if ctx.get("price_item") else ("price_ref" if ctx.get("price_ref") else "catalog"),
                "matched_service_id": ctx["service_id"],
                "service": ctx["service"],
                "match_score": 1.0,
                "is_confident": True,
                "price_key": ctx.get("price_key"),
                "price_ref": ctx.get("price_ref"),
                "price_item": ctx.get("price_item"),
                "context_doc_id": ctx.get("context_doc_id"),
                "fallback_reason": "context_session",
            }
        return {
            "mode": "clarify",
            "intent": intent,
            "fallback_reason": "service_not_found",
            **match,
        }
    if not match.get("is_confident"):
        ctx = _service_from_session_context(sid, client_id)
        if ctx and intent == "price_lookup":
            return {
                "mode": "matched",
                "intent": intent,
                "route_source": "prices_json" if ctx.get("price_item") else ("price_ref" if ctx.get("price_ref") else "catalog"),
                "matched_service_id": ctx["service_id"],
                "service": ctx["service"],
                "match_score": 1.0,
                "is_confident": True,
                "price_key": ctx.get("price_key"),
                "price_ref": ctx.get("price_ref"),
                "price_item": ctx.get("price_item"),
                "context_doc_id": ctx.get("context_doc_id"),
                "fallback_reason": "context_session",
            }
        return {
            "mode": "clarify",
            "intent": intent,
            "fallback_reason": "low_match_score",
            **match,
        }
    prices = _read_json_dict(_client_json_path(client_id, "prices.json"))
    service = match.get("service") or {}
    price_ref = service.get("price_ref")
    price_key = service.get("price_key")
    price_item = prices.get(price_key) if isinstance(prices, dict) and price_key else None
    route_source = "catalog"
    if intent == "price_concern":
        route_source = "catalog"
    elif price_ref:
        route_source = "price_ref"
    elif price_item is not None:
        route_source = "prices_json"
    return {
        "mode": "matched",
        "intent": intent,
        "route_source": route_source,
        "price_key": price_key,
        "price_ref": price_ref,
        "price_item": price_item if isinstance(price_item, dict) else None,
        **match,
    }


def select_catalog_content_route(q: str, *, client_id: str | None) -> dict:
    """Информационный маршрут по service_catalog (без ценового интента).

    Только для сервисов без MD-страницы (md_entry_ref=null) с facts-карточкой.
    Контентные запросы к сервисам с MD-страницей идут в retrieval — он семантически
    точнее и не требует эвристик для разграничения запросов. Такой подход
    масштабируется на любой клиентский корпус без изменения логики роутинга.
    """
    match = match_service_from_catalog(q, client_id=client_id)
    if not match.get("matched_service_id") or not match.get("is_confident"):
        return {"mode": "none"}
    service = match.get("service") or {}
    md_raw = service.get("md_entry_ref")
    if isinstance(md_raw, str) and md_raw.strip():
        return {"mode": "none"}
    facts = [str(x).strip() for x in (service.get("facts") or []) if str(x).strip()]
    if not facts:
        return {"mode": "none"}
    return {
        "mode": "facts",
        "matched_service_id": match.get("matched_service_id"),
        "match_score": match.get("match_score"),
        "service": service,
    }
