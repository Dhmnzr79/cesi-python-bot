"""Промпты и вызовы OpenAI (чат); эмпатия."""
import json
import os
import re

from openai import OpenAI

from config import (
    BOOKING_INTENT_LLM_MODEL,
    BOOKING_INTENT_LLM_ON,
    CHAT_JSON_MODE,
    CHAT_MODEL,
    EMPATHY_ON,
    LEAD_NAME_CLASSIFY_MODEL,
    MEMORY_ON,
    OPENAI_API_KEY,
    PRICE_INTENT_LLM_MODEL,
    PRICE_INTENT_LLM_ON,
    QUERY_REWRITE_MAX_MESSAGES,
    QUERY_REWRITE_MODEL,
    QUERY_REWRITE_ON,
    QUERY_REWRITE_VALIDATE_OVERLAP,
    REWRITE_REJECT_SUBSTRINGS,
    TRIGGERS_COMPILED,
)
from logging_setup import get_logger, log_json
from session import (
    is_first_in_topic,
    mem_context,
    mem_get,
    update_topic_empathy,
)

client = OpenAI(api_key=OPENAI_API_KEY)
logger = get_logger("bot")
LLM_REQUEST_TIMEOUT_SEC = float(os.getenv("LLM_REQUEST_TIMEOUT_SEC", "20"))
LLM_FALLBACK_ANSWER = os.getenv(
    "LLM_FALLBACK_ANSWER",
    "Извините, сейчас есть техническая задержка. Могу повторить ответ или предложить консультацию.",
)

_REWRITE_SYSTEM = (
    "Ты формулируешь поисковый запрос для семантического поиска по базе знаний стоматологии. "
    "По последним репликам диалога и текущему вопросу пациента напиши одну короткую строку на русском "
    "для векторного поиска (ключевые сущности: врач, процедура, симптом, зуб, материал). "
    "Не выдумывай факты: опирайся только на явное в диалоге и в текущем вопросе. "
    "Если вопрос уже самодостаточен — сожми до сути без лишних слов. "
    'Ответь одним JSON-объектом с ключом "search_query" (строка). Без markdown.'
)


def _norm_rewrite_compare(s: str) -> str:
    x = (s or "").strip().lower().replace("ё", "е")
    x = re.sub(r"[^\w\s\-]", " ", x, flags=re.U)
    return re.sub(r"\s+", " ", x).strip()


def validated_retrieval_rewrite(q_user: str, model_out: str) -> tuple[str, str | None]:
    """Вернуть (эффективная строка для доп. семантики, причина отказа или None).

    Эффективная строка никогда не бывает пустой при непустом q_user."""
    u0 = (q_user or "").strip()
    w0 = (model_out or "").strip()
    if not u0:
        return w0, None
    if not w0 or w0.lower() == u0.lower():
        return u0 if not w0 else w0, None

    wl = w0.lower()
    for marker in REWRITE_REJECT_SUBSTRINGS:
        if marker and marker in wl:
            return u0, "prompt_leak"

    if QUERY_REWRITE_VALIDATE_OVERLAP and not _rewrite_overlaps_user_question(u0, w0):
        return u0, "no_overlap"

    return w0, None


def _rewrite_overlaps_user_question(q_user: str, q_rewrite: str) -> bool:
    """Есть ли общая содержательная связь между исходным вопросом и переписанным запросом."""
    u = _norm_rewrite_compare(q_user)
    r = _norm_rewrite_compare(q_rewrite)
    if not u or not r:
        return True
    for tok in u.split():
        if len(tok) >= 4 and tok[:4] in r:
            return True
        if 3 <= len(tok) < 4 and tok in r.split():
            return True
    for tok in r.split():
        if len(tok) >= 4 and tok[:4] in u:
            return True
        if 3 <= len(tok) < 4 and tok in u.split():
            return True
    return False


def rewrite_query_for_retrieval(
    session_id: str, current_q: str, *, client_id: str | None = None
) -> str:
    """Переписать вопрос для retrieval с учётом последних реплик (текущий ход ещё не в hist)."""
    q0 = (current_q or "").strip()
    if not QUERY_REWRITE_ON or not q0:
        return q0
    st = mem_get(session_id)
    hist = list(st.get("hist") or [])
    if not hist:
        return q0
    tail = hist[-QUERY_REWRITE_MAX_MESSAGES:]
    dialog_lines = [f"{m.get('role', '?')}: {m.get('content', '')}" for m in tail]
    dialog_block = "\n".join(dialog_lines)
    user_block = (
        "Последние реплики диалога:\n"
        f"{dialog_block}\n\n"
        "Текущий вопрос пациента:\n"
        f"{q0}"
    )
    try:
        resp = client.chat.completions.create(
            model=QUERY_REWRITE_MODEL,
            temperature=0.15,
            max_completion_tokens=200,
            response_format={"type": "json_object"},
            timeout=LLM_REQUEST_TIMEOUT_SEC,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": user_block},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("rewrite_not_object")
        sq = obj.get("search_query")
        if sq is None and "query" in obj:
            sq = obj.get("query")
        out = str(sq).strip() if sq is not None else ""
        if not out or len(out) > 600:
            raise ValueError("rewrite_empty_or_long")
        effective, reject_reason = validated_retrieval_rewrite(q0, out)
        if reject_reason:
            log_json(
                logger,
                "retrieval_query_rewrite_rejected",
                client_id=client_id,
                sid=session_id,
                model_used=QUERY_REWRITE_MODEL,
                query_raw=q0[:200],
                model_out=out[:200],
                reason=reject_reason,
                effective=effective[:200],
            )
        rewrite_applied = effective.lower() != q0.lower()
        log_json(
            logger,
            "retrieval_query_rewrite",
            client_id=client_id,
            sid=session_id,
            model_used=QUERY_REWRITE_MODEL,
            query_raw=q0[:200],
            query_for_retrieval=effective[:200],
            rewrite_applied=rewrite_applied,
            model_raw_before_validate=out[:200] if reject_reason else None,
        )
        return effective
    except Exception as e:
        log_json(
            logger,
            "retrieval_query_rewrite_failed",
            client_id=client_id,
            sid=session_id,
            model_used=QUERY_REWRITE_MODEL,
            query_raw=q0[:200],
            err=str(e)[:300],
        )
        return q0


_FACTS_CARD_SYSTEM = (
    "Ты помощник стоматологической клиники. "
    "Тебе дан вопрос пациента, название услуги и список фактов о ней. "
    "Напиши живой разговорный ответ — 2-3 предложения. "
    "Правила: ответь именно на вопрос пациента (если спрашивает 'делаете ли?' — сначала подтверди одним словом); "
    "используй ТОЛЬКО факты из списка, ничего не добавляй от себя; "
    "все цифры и числовые показатели из фактов обязательно сохрани; "
    "не перечисляй факты списком — пиши текстом; "
    "тон спокойный и доброжелательный, без канцелярита. "
    'Ответь одним JSON-объектом с ключом "answer".'
)


def generate_facts_card_answer(
    title: str,
    facts: list[str],
    *,
    sid: str,
    client_id: str | None,
    user_question: str = "",
) -> str | None:
    if not facts:
        return None
    facts_block = "\n".join(f"- {f}" for f in facts)
    q_line = f"Вопрос пациента: {user_question}\n\n" if user_question else ""
    user_msg = f"{q_line}Услуга: {title}\n\nФакты:\n{facts_block}"
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            max_completion_tokens=300,
            response_format={"type": "json_object"},
            timeout=LLM_REQUEST_TIMEOUT_SEC,
            messages=[
                {"role": "system", "content": _FACTS_CARD_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        answer = str(obj.get("answer") or "").strip()
        if answer:
            log_json(logger, "facts_card_llm", client_id=client_id, sid=sid, title=title)
            return answer
    except Exception as exc:
        log_json(logger, "facts_card_llm_error", client_id=client_id, sid=sid, error=str(exc))
    return None


BASE_SYSTEM = (
    "Ты — спокойный и доброжелательный врач-имплантолог. "
    "Отвечай кратко, с эмпатией и только по предоставленному контенту, но не слепо цитируй контент."
    "Если информации нет — мягко скажи об этом и предложи консультацию."
)

EMPATHY_ADDON = (
    " Добавь 1 короткое предложение эмпатии в начале или конце ответа, "
    "строго по тону вопроса пациента (страх/боль, безопасность, стоимость, сроки, подходит ли). "
    "Эмпатия должна быть естественной, без клише и без повторения фактов. "
    "После неё дай точный ответ по контенту."
)

JSON_ANSWER_RULE = (
    ' Ответь одним JSON-объектом с единственным ключом "answer" (строка с текстом для пациента). '
    "Без markdown, без пояснений вне JSON."
)


def _norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е").strip()


def _doc_key(md_file: str, meta: dict) -> str:
    return meta.get("doc_id") or md_file


def _is_emotional(user_q: str, empathy_tag: str | None) -> bool:
    q = _norm(user_q)
    if empathy_tag and empathy_tag in TRIGGERS_COMPILED:
        if TRIGGERS_COMPILED[empathy_tag].search(q):
            return True
    for rx in TRIGGERS_COMPILED.values():
        if rx.search(q):
            return True
    return False


def build_messages_for_gpt(user_q: str, context_md: str, meta: dict, session_id: str):
    doc_key = _doc_key(
        meta.get("md_file") or meta.get("source") or meta.get("title", ""),
        meta,
    )
    allow_empathy = bool(EMPATHY_ON and meta.get("empathy_enabled"))
    first_in_topic = is_first_in_topic(session_id, doc_key)
    emotional = _is_emotional(user_q, meta.get("empathy_tag"))

    use_empathy = bool(allow_empathy and (first_in_topic or emotional))
    system_prompt = BASE_SYSTEM + (EMPATHY_ADDON if use_empathy else "")
    if meta.get("verbatim"):
        system_prompt += (
            " Используй только точные формулировки из предоставленного контента, "
            "не перефразируй."
        )
    pref_fmt = meta.get("preferred_format") or []
    if isinstance(pref_fmt, str):
        pref_fmt = [pref_fmt]
    pref_fmt = [str(x).strip().lower() for x in pref_fmt if str(x).strip()]
    if "bullets" in pref_fmt:
        system_prompt += " Структурируй ответ в виде коротких пунктов."
    elif "paragraph" in pref_fmt:
        system_prompt += " Отвечай связным текстом без списков."
    if CHAT_JSON_MODE:
        system_prompt += JSON_ANSWER_RULE

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Вопрос пациента:\n"
                + user_q.strip()
                + "\n\nКонтент для ответа (markdown, цитируй по смыслу, не выдумывай):\n"
                + context_md.strip()
            ),
        },
    ]

    meta["_empathy_used"] = use_empathy
    meta["_first_in_topic"] = first_in_topic
    meta["_emotional_detected"] = emotional
    meta["_doc_key"] = doc_key

    return messages, use_empathy, doc_key


def generate_answer_with_empathy(
    user_q: str, context_md: str, meta: dict, session_id: str
) -> tuple[str, dict]:
    mem_txt, profile = mem_context(session_id)

    messages, use_empathy, doc_key = build_messages_for_gpt(
        user_q, context_md, meta, session_id
    )

    if mem_txt and MEMORY_ON:
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] = f"{mem_txt}\n\n" + msg["content"]
                break

    kwargs = dict(model=CHAT_MODEL, temperature=0.3, messages=messages)
    if CHAT_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}
    kwargs["timeout"] = LLM_REQUEST_TIMEOUT_SEC
    try:
        resp = client.chat.completions.create(**kwargs)
        raw = (resp.choices[0].message.content or "").strip()
        answer = raw
        if CHAT_JSON_MODE:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and obj.get("answer"):
                    answer = str(obj["answer"]).strip()
            except (json.JSONDecodeError, TypeError):
                pass
        if not (answer or "").strip():
            answer = LLM_FALLBACK_ANSWER
        log_json(
            logger,
            "llm_generate",
            sid=session_id,
            model_used=CHAT_MODEL,
            empathy_used=bool(use_empathy),
            used_fallback=bool(answer == LLM_FALLBACK_ANSWER),
        )
    except Exception as e:
        log_json(
            logger,
            "llm_generate_failed",
            sid=session_id,
            model_used=CHAT_MODEL,
            err=str(e)[:300],
        )
        answer = LLM_FALLBACK_ANSWER

    update_topic_empathy(session_id, doc_key, use_empathy)

    return answer, profile


_NAME_CLASSIFY_SYSTEM = (
    "Ты классификатор короткой строки на шаге «как к вам обращаться» в чате стоматологии. "
    "Нужно решить, пригодна ли строка как личное обращение к человеку.\n"
    "Значения label:\n"
    "- valid_name — нормальное имя или обращение (имя, имя и отчество, имя и фамилия, "
    "в т.ч. латиница вроде Kai Chen).\n"
    "- invalid_name — явно не имя: вопрос по клинике/лечению, оскорбление или псевдо-фамилия для троллинга, "
    "служебный текст вместо имени.\n"
    "- unsure — формально похоже на имя (1–3 коротких слова), но смысл неоднозначен: ник, шутка, "
    "нарицательное слово как обращение (например «Рыба», «Лиса»).\n"
    'Ответь одним JSON-объектом с ключом "label" и значением ровно одним из: '
    '"valid_name", "invalid_name", "unsure". Без markdown и текста вне JSON.'
)


def classify_lead_name_shape(
    candidate: str, raw_user: str, *, client_id: str | None, sid: str
) -> str:
    """Только для строк, прошедших жёсткий предфильтр и extract_name."""
    c = (candidate or "").strip()
    r = (raw_user or "").strip()
    if not c:
        return "invalid_name"
    payload = json.dumps({"candidate": c, "original": r}, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=LEAD_NAME_CLASSIFY_MODEL,
            temperature=0,
            max_completion_tokens=60,
            response_format={"type": "json_object"},
            timeout=LLM_REQUEST_TIMEOUT_SEC,
            messages=[
                {"role": "system", "content": _NAME_CLASSIFY_SYSTEM},
                {"role": "user", "content": payload},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("name_classify_not_object")
        label = str(obj.get("label") or "").strip().lower()
        if label in ("valid_name", "invalid_name", "unsure"):
            log_json(
                logger,
                "lead_name_classify",
                client_id=client_id,
                sid=sid,
                label=label,
                candidate=c[:80],
            )
            return label
    except Exception as e:
        log_json(
            logger,
            "lead_name_classify_failed",
            client_id=client_id,
            sid=sid,
            err=str(e)[:300],
            candidate=c[:80],
        )
    return "unsure"


_BOOKING_INTENT_SYSTEM = (
    "Ты классификатор намерения в чате стоматологии. Пользователь только что написал одну реплику.\n"
    "wants_booking = true, если он явно хочет записаться на приём/консультацию, оставить заявку на связь, "
    "попросить записать его сейчас (в т.ч. с опечатками: «записатся», «зописаться», «хачу записаться»).\n"
    "wants_booking = false, если это вопрос по лечению, ценам, FAQ «как записаться / куда звонить», "
    "общая консультация без явной просьбы записать именно его, или просто болтовня.\n"
    'Ответь одним JSON-объектом с ключом "wants_booking" (boolean true или false). '
    "Без markdown и текста вне JSON."
)


def classify_booking_wants_appointment(
    user_message: str, *, client_id: str | None, sid: str
) -> bool:
    if not BOOKING_INTENT_LLM_ON:
        return False
    msg = (user_message or "").strip()
    if len(msg) < 2:
        return False
    try:
        resp = client.chat.completions.create(
            model=BOOKING_INTENT_LLM_MODEL,
            temperature=0,
            max_completion_tokens=40,
            response_format={"type": "json_object"},
            timeout=LLM_REQUEST_TIMEOUT_SEC,
            messages=[
                {"role": "system", "content": _BOOKING_INTENT_SYSTEM},
                {"role": "user", "content": msg[:600]},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("booking_intent_not_object")
        wb = obj.get("wants_booking")
        out = wb is True or str(wb).lower() in ("true", "1", "yes")
        log_json(
            logger,
            "booking_intent_llm",
            client_id=client_id,
            sid=sid,
            wants_booking=out,
            msg_len=len(msg),
        )
        return out
    except Exception as e:
        log_json(
            logger,
            "booking_intent_llm_failed",
            client_id=client_id,
            sid=sid,
            err=str(e)[:300],
        )
        return False


_PRICE_INTENT_SYSTEM = (
    "Ты классификатор ценового намерения в чате стоматологии. "
    "Нужно выбрать один label: "
    "price_lookup (пользователь спрашивает цену/стоимость конкретной услуги), "
    "price_concern (сомнение или возражение по цене: дорого, почему так дорого, не по карману), "
    "other (неценовой вопрос). "
    "Важно: вопросы про скидки, полис ОМС/ДМС, рассрочку, оплату по частям без жалобы «дорого» — это other. "
    'Ответь одним JSON-объектом: {"label":"price_lookup|price_concern|other"}. '
    "Без markdown и текста вне JSON."
)


def classify_price_intent(user_message: str, *, client_id: str | None, sid: str) -> str:
    if not PRICE_INTENT_LLM_ON:
        return "other"
    msg = (user_message or "").strip()
    if len(msg) < 2:
        return "other"
    try:
        resp = client.chat.completions.create(
            model=PRICE_INTENT_LLM_MODEL,
            temperature=0,
            max_completion_tokens=50,
            response_format={"type": "json_object"},
            timeout=LLM_REQUEST_TIMEOUT_SEC,
            messages=[
                {"role": "system", "content": _PRICE_INTENT_SYSTEM},
                {"role": "user", "content": msg[:700]},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("price_intent_not_object")
        label = str(obj.get("label") or "").strip().lower()
        if label not in {"price_lookup", "price_concern", "other"}:
            label = "other"
        log_json(
            logger,
            "price_intent_llm",
            client_id=client_id,
            sid=sid,
            label=label,
            msg_len=len(msg),
        )
        return label
    except Exception as e:
        log_json(
            logger,
            "price_intent_llm_failed",
            client_id=client_id,
            sid=sid,
            err=str(e)[:300],
        )
        return "other"
