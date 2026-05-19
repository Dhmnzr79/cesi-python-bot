"""A3.3 deterministic doctor index (frontmatter + catalog)."""
from __future__ import annotations

import glob
import json
import os
import re
import threading
from typing import Any, TypedDict

import alias_lexical
import yaml

from query_selector import match_service_from_catalog

_MD_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "md")

_NAMES_INDEX_LOCK = threading.Lock()
_NAMES_INDEX: dict[str, tuple[float, frozenset[str]]] = {}

_GROUND_TRUTH_LOCK = threading.Lock()
_GROUND_TRUTH_INDEX: dict[str, tuple[float, frozenset[str], frozenset[str]]] = {}

# sentinel: в запросе нет ключевых слов из SPECIALTY_KEYWORDS
_NO_SPECIALTY_KEYWORD = object()
_OVERVIEW_ID = "doctors__doctor__overview"

# Ключевое слово → topic (как префикс в md_entry_ref услуги или поле topic в каталоге при появлении).
SPECIALTY_KEYWORDS: dict[str, str | None] = {
    "имплантолог": "implantation",
    "ортопед": "prosthetics",
    "протезист": "prosthetics",
    "терапевт": None,
    "хирург": None,
    "ортодонт": "orthodontics",
}


class DoctorPublic(TypedDict, total=False):
    doc_id: str
    path: str
    name_full: str
    name_short: str
    position: str
    experience_years: int
    services: list[str]
    aliases: list[str]
    brief: str


class DoctorListFact(TypedDict, total=False):
    """Structured facts для LLM-компоновки списка врачей (без шаблонной склейки)."""

    doc_id: str  # routing / дедуп; не показывается пациенту
    name_full: str
    name_short: str
    position: str
    experience_years: int  # только если есть во frontmatter
    specialty_brief: str


def _safe_client_id(client_id: str | None) -> str:
    return (client_id or "default").strip() or "default"


def _client_catalog_path(client_id: str | None) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "clients", _safe_client_id(client_id), "service_catalog.json")


def _read_service_catalog(client_id: str | None) -> dict[str, Any]:
    path = _client_catalog_path(client_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _infer_entry_topic(entry: dict[str, Any]) -> str | None:
    t = entry.get("topic")
    if isinstance(t, str) and t.strip():
        return t.strip().lower()
    ref = str(entry.get("md_entry_ref") or "").strip().lower().removesuffix(".md")
    if not ref:
        return None
    first = ref.split("__", 1)[0].strip()
    return first or None


def _services_list(fm: dict[str, Any]) -> list[str]:
    raw = fm.get("services")
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _read_md_split(path: str) -> tuple[dict[str, Any], str, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    base = os.path.basename(path)
    doc_stem = base[:-3] if base.lower().endswith(".md") else base
    if not raw.lstrip().startswith("---"):
        return {}, raw, doc_stem
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw, doc_stem
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        fm = {}
    return (fm if isinstance(fm, dict) else {}), parts[2], doc_stem


def _h1(body: str) -> str:
    m = re.search(r"^##\s+(.+)$", body, flags=re.M)
    return (m.group(1) or "").strip() if m else ""


def _korotko_paragraph(body: str) -> str:
    """Текст секции #korotko без обрезки по числу символов."""
    m = re.search(
        r"^###\s+Коротко.*?\{#korotko\}\s*\n([\s\S]*?)(?=^#{1,6}\s|\Z)",
        body,
        flags=re.M | re.I,
    )
    if not m:
        m = re.search(r"^###\s+Коротко\s*\n([\s\S]*?)(?=^#{1,6}\s|\Z)", body, flags=re.M | re.I)
    chunk = (m.group(1) if m else body).strip()
    return re.sub(r"\s+", " ", chunk, flags=re.U)


def _korotko_specialty_brief(body: str) -> str:
    """Первая фраза из #korotko: до первой точки или весь абзац; без лимита символов."""
    chunk = _korotko_paragraph(body).strip()
    if not chunk:
        return ""
    dot = chunk.find(".")
    if dot != -1:
        return chunk[: dot + 1].strip()
    return chunk


def _korotko_brief(body: str, *, limit: int = 220) -> str:
    """Лимит символов для DoctorPublic.brief (путь по имени врача и пр.); не использовать для списков врачей."""
    chunk = _korotko_paragraph(body)
    return (chunk[:limit] + "…") if len(chunk) > limit else chunk


def _experience_years_optional(fm: dict[str, Any]) -> int | None:
    if "experience_years" not in fm:
        return None
    raw = fm.get("experience_years")
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    try:
        v = int(float(str(raw).strip().replace(",", ".")))
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v


def _lemma_set(text: str) -> set[str]:
    q = (text or "").lower().replace("ё", "е")
    q = re.sub(r"[^\w\s]", " ", q, flags=re.U)
    toks = [t for t in q.split() if len(t) >= 2]
    return set(alias_lexical.lemma_forms_for_tokens(toks))


def _name_hit_score(q_lem: set[str], phrase: str) -> float:
    p_lem = _lemma_set(phrase)
    if not p_lem or not q_lem:
        return 0.0
    if p_lem <= q_lem:
        return 1.0
    inter = len(q_lem & p_lem)
    if inter == 0:
        return 0.0
    return inter / max(len(p_lem), 1)


def _name_candidates_from_fm(fm: dict[str, Any], h1: str) -> list[str]:
    phrases: list[str] = []
    for k in ("name_short", "name_full"):
        v = str(fm.get(k) or "").strip()
        if v:
            phrases.append(v)
    for a in fm.get("aliases") or []:
        s = str(a).strip()
        if s:
            phrases.append(s)
    if h1.strip():
        phrases.append(h1.strip())
    return phrases


def _doctor_record(path: str, fm: dict[str, Any], body: str, stem: str) -> DoctorPublic:
    doc_id = str(fm.get("doc_id") or stem).strip().removesuffix(".md")
    h1 = _h1(body)
    name_full = str(fm.get("name_full") or h1 or doc_id).strip()
    name_short = str(fm.get("name_short") or "").strip()
    if not name_short and name_full:
        parts = name_full.replace("ё", "е").split()
        if parts:
            name_short = parts[0].strip(",.")
    svc = _services_list(fm)
    pos = str(fm.get("position") or "").strip()
    exp_opt = _experience_years_optional(fm)
    aliases = [str(x).strip() for x in (fm.get("aliases") or []) if str(x).strip()]
    brief = _korotko_brief(body)
    rec: DoctorPublic = {
        "doc_id": doc_id,
        "path": path,
        "name_full": name_full,
        "position": pos,
        "services": svc,
        "aliases": aliases,
        "brief": brief,
    }
    if name_short:
        rec["name_short"] = name_short
    if exp_opt is not None:
        rec["experience_years"] = exp_opt
    return rec


def _doctor_list_fact(_path: str, fm: dict[str, Any], body: str, stem: str) -> DoctorListFact:
    doc_id = str(fm.get("doc_id") or stem).strip().removesuffix(".md")
    h1 = _h1(body)
    name_full = str(fm.get("name_full") or h1 or doc_id).strip()
    name_short = str(fm.get("name_short") or "").strip()
    if not name_short and name_full:
        parts = name_full.replace("ё", "е").split()
        if parts:
            name_short = parts[0].strip(",.")
    pos = str(fm.get("position") or "").strip()
    spec = _korotko_specialty_brief(body)
    fact: DoctorListFact = {
        "doc_id": doc_id,
        "name_full": name_full,
        "position": pos,
        "specialty_brief": spec,
    }
    if name_short:
        fact["name_short"] = name_short
    exp_opt = _experience_years_optional(fm)
    if exp_opt is not None:
        fact["experience_years"] = exp_opt
    return fact


def _iter_doctor_paths() -> list[str]:
    return sorted(
        p
        for p in glob.glob(os.path.join(_MD_BASE, "doctors__doctor__*.md"))
        if os.path.basename(p).lower() != "doctors__doctor__overview.md"
    )


def _doctor_paths_mtime_max() -> float:
    m = 0.0
    for p in _iter_doctor_paths():
        try:
            m = max(m, os.path.getmtime(p))
        except OSError:
            continue
    return m


def _collect_client_doctor_name_phrases() -> frozenset[str]:
    """Подстроковые ключи из frontmatter + H1 активных врачей (нижний регистр, ё→е)."""
    phrases: set[str] = set()
    for path in _iter_doctor_paths():
        fm, body, stem = _read_md_split(path)
        if fm.get("active") is False:
            continue
        h1 = _h1(body)
        for raw in _name_candidates_from_fm(fm, h1):
            t = raw.strip().lower().replace("ё", "е")
            t = re.sub(r"\s+", " ", t, flags=re.U).strip()
            if len(t) >= 2:
                phrases.add(t)
    return frozenset(phrases)


def cached_doctor_name_substrings(*, client_id: str | None) -> frozenset[str]:
    """Кэш по client_id и mtime md врачей."""
    cid = _safe_client_id(client_id)
    mt = _doctor_paths_mtime_max()
    with _NAMES_INDEX_LOCK:
        hit = _NAMES_INDEX.get(cid)
        if hit is not None and hit[0] == mt:
            return hit[1]
        phrases = _collect_client_doctor_name_phrases()
        _NAMES_INDEX[cid] = (mt, phrases)
        return phrases


def _norm_ground_truth_text(text: str) -> str:
    return (text or "").strip().lower().replace("ё", "е")


def _build_doctor_ground_truth_index() -> tuple[frozenset[str], frozenset[str]]:
    """Role phrases from position/aliases + specialty keys confirmed by doctor md."""
    role_phrases: set[str] = set()
    confirmed_kw: set[str] = set()
    for path in _iter_doctor_paths():
        fm, _body, _stem = _read_md_split(path)
        if fm.get("active") is False:
            continue
        pos = _norm_ground_truth_text(str(fm.get("position") or ""))
        alias_blob = " ".join(
            _norm_ground_truth_text(str(a)) for a in (fm.get("aliases") or []) if str(a).strip()
        )
        blob = f"{pos} {alias_blob}".strip()
        if "главный врач" in pos:
            role_phrases.add("главный врач")
        if pos:
            for part in re.split(r"[,;]", pos):
                for frag in re.split(r"[-–]", part):
                    f = re.sub(r"\s+", " ", frag).strip()
                    if len(f) >= 5:
                        role_phrases.add(f)
        for kw in SPECIALTY_KEYWORDS:
            if kw in blob:
                confirmed_kw.add(kw)
    return frozenset(role_phrases), frozenset(confirmed_kw)


def cached_doctor_ground_truth_index(
    *, client_id: str | None
) -> tuple[frozenset[str], frozenset[str]]:
    """(role_phrases, confirmed_specialty_keywords) cached by md mtime."""
    cid = _safe_client_id(client_id)
    mt = _doctor_paths_mtime_max()
    with _GROUND_TRUTH_LOCK:
        hit = _GROUND_TRUTH_INDEX.get(cid)
        if hit is not None and hit[0] == mt:
            return hit[1], hit[2]
        role_phrases, confirmed_kw = _build_doctor_ground_truth_index()
        _GROUND_TRUTH_INDEX[cid] = (mt, role_phrases, confirmed_kw)
        return role_phrases, confirmed_kw


def catalog_has_active_topic(topic: str, *, client_id: str | None) -> bool:
    """True if service_catalog has an active entry for this topic prefix."""
    tnorm = str(topic or "").strip().lower()
    if not tnorm:
        return False
    catalog = _read_service_catalog(client_id)
    for _sid, entry in catalog.items():
        if not isinstance(entry, dict) or entry.get("active") is False:
            continue
        et = _infer_entry_topic(entry)
        if et == tnorm:
            return True
    return False


def doctor_ground_truth_mention(text: str, *, client_id: str | None) -> bool:
    """
    True if question mentions a doctor name, confirmed role, or specialty backed by
    doctors md and/or active catalog topic (ingress ground truth only; not routing).
    """
    low = _norm_ground_truth_text(text)
    if not low:
        return False
    for phrase in cached_doctor_name_substrings(client_id=client_id):
        if len(phrase) >= 3 and phrase in low:
            return True
    role_phrases, confirmed_kw = cached_doctor_ground_truth_index(client_id=client_id)
    for phrase in role_phrases:
        if len(phrase) >= 4 and phrase in low:
            return True
    for kw, topic in SPECIALTY_KEYWORDS.items():
        if kw not in low:
            continue
        if kw in confirmed_kw:
            return True
        if topic and catalog_has_active_topic(topic, client_id=client_id):
            return True
    return False


def load_all_doctors() -> list[DoctorPublic]:
    """Все активные записи докторских md (без overview)."""
    out: list[DoctorPublic] = []
    for path in _iter_doctor_paths():
        fm, body, stem = _read_md_split(path)
        if fm.get("active") is False:
            continue
        out.append(_doctor_record(path, fm, body, stem))
    return out


def find_doctors_by_service(service_id: str, *, client_id: str) -> list[DoctorListFact]:
    sid = str(service_id or "").strip()
    if not sid:
        return []
    found: list[DoctorListFact] = []
    for path in _iter_doctor_paths():
        fm, body, stem = _read_md_split(path)
        if fm.get("active") is False:
            continue
        svc = _services_list(fm)
        if sid in svc:
            found.append(_doctor_list_fact(path, fm, body, stem))
    found.sort(key=lambda d: str(d.get("name_full") or d.get("doc_id") or ""))
    return found


def find_doctors_by_topic(topic: str, *, client_id: str) -> list[DoctorListFact]:
    tnorm = str(topic or "").strip().lower()
    if not tnorm:
        return []
    catalog = _read_service_catalog(client_id)
    service_ids: set[str] = set()
    for sid, entry in catalog.items():
        if not isinstance(entry, dict) or not bool(entry.get("active", True)):
            continue
        et = _infer_entry_topic(entry)
        if et == tnorm:
            service_ids.add(str(sid))
    if not service_ids:
        return []
    found: list[DoctorListFact] = []
    seen: set[str] = set()
    for path in _iter_doctor_paths():
        fm, body, stem = _read_md_split(path)
        if fm.get("active") is False:
            continue
        svc = set(_services_list(fm))
        if not (svc & service_ids):
            continue
        rec = _doctor_list_fact(path, fm, body, stem)
        did = str(rec.get("doc_id") or "")
        if did and did not in seen:
            seen.add(did)
            found.append(rec)
    found.sort(key=lambda d: str(d.get("name_full") or d.get("doc_id") or ""))
    return found


def _specialty_topic_from_query(q_raw: str) -> str | None | object:
    """topic str | None (терапевт/хирург → overview) | _NO_SPECIALTY_KEYWORD."""
    low = (q_raw or "").lower()
    for kw in sorted(SPECIALTY_KEYWORDS.keys(), key=len, reverse=True):
        if kw in low:
            return SPECIALTY_KEYWORDS[kw]
    return _NO_SPECIALTY_KEYWORD


def _norm_query(q: str) -> str:
    x = (q or "").strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", x, flags=re.U).strip()


def _is_generic_team_question(q_raw: str) -> bool:
    low = _norm_query(q_raw)
    if "врач" not in low and "доктор" not in low and "специалист" not in low:
        return False
    return bool(
        re.search(
            r"\b(кто|какие|расскаж|покаж|наш|работа|команда|есть\s+ли|сколько)\b",
            low,
            flags=re.I,
        )
    )


def _dedupe_doctor_facts(recs: list[DoctorListFact]) -> list[DoctorListFact]:
    seen: set[str] = set()
    out: list[DoctorListFact] = []
    for r in recs:
        did = str(r.get("doc_id") or "")
        if not did or did in seen:
            continue
        seen.add(did)
        out.append(r)
    return out


def doctor_list_fact_public_dict(f: DoctorListFact) -> dict[str, Any]:
    """Поля для LLM-контекста (без doc_id)."""
    out: dict[str, Any] = {
        "name_full": f.get("name_full") or "",
        "position": f.get("position") or "",
        "specialty_brief": f.get("specialty_brief") or "",
    }
    if f.get("name_short"):
        out["name_short"] = f["name_short"]
    if f.get("experience_years") is not None:
        out["experience_years"] = f["experience_years"]
    return out


def build_doctors_list_llm_question(*, user_question: str) -> str:
    q0 = (user_question or "").strip()
    rules = (
        "Перечисли врачей, которые делают эту услугу. Для каждого укажи: полное имя, должность; "
        "если в данных есть experience_years — добавь кратко стаж в годах; "
        "затем одно короткое предложение про подход по полю specialty_brief.\n"
        "Заверши приглашением на бесплатную консультацию.\n"
        "Используй ТОЛЬКО факты ниже, ничего не выдумывай. "
        "Если для врача нет experience_years в данных — не упоминай его стаж и не подставляй чужие числа; "
        "не используй слово «null»."
    )
    if q0:
        return f"Вопрос пациента: {q0}\n\n{rules}"
    return rules


def build_synthetic_doctors_list_chunk(
    *,
    client_id: str | None,
    facts: list[DoctorListFact],
) -> dict[str, Any]:
    """Один «источник» для Generator: JSON с фактами (корпусный chunk-объект)."""
    public = [doctor_list_fact_public_dict(f) for f in facts]
    body = json.dumps(public, ensure_ascii=False, indent=2)
    text = (
        "РОЛЬ: DOCTORS_LIST — составь ответ по инструкции из вопроса.\n"
        "Структурированные факты (JSON, используй только их):\n"
        + body
    )
    return {
        "file": "doctors__doctor__overview.md",
        "h2": "",
        "h3": "",
        "h2_id": None,
        "h3_id": "doctors_list",
        "text": text,
        "_score": 1.0,
        "client_id": client_id,
    }


def doctor_intent_probe(q: str) -> bool:
    """Позитивное намерение «про врачей / кто из персонала»; без цены / процесса / «как врач …»."""
    x = _norm_query(q)
    if len(x) < 3:
        return False

    if re.search(r"\bсколько\s+стоит\b", x):
        return False
    if re.search(r"\bсколько\s+длится\b", x):
        return False
    if "противопоказ" in x:
        return False
    if re.search(r"\bбольно\s+ли\b", x):
        return False
    if re.search(r"\bкак\s+проходит\b", x):
        return False
    if re.search(r"\bкак\s+врач\b", x):
        return False
    if re.search(r"\bкак\s+доктор\b", x):
        return False

    for role in sorted(SPECIALTY_KEYWORDS.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(role) + r"\b", x):
            return True

    if re.search(
        r"\b(?:кто|какой|какая|какие)(?:\s+у\s+вас)?\s+(?:ваши?\s+)?(?:врач|врачи|доктор|доктора|специалист|специалисты)\b",
        x,
    ):
        return True

    if re.search(
        r"\b(?:кто|какой|какая|какие)\s+(?:делает|делают|ставит|ставят|ведёт|ведет|ведут|принимает|принимают|занимается|занимаются)\b",
        x,
    ):
        return True

    if re.search(
        r"\b(?:какие|какой)\s+(?:врачи|специалисты)\s+(?:у\s+вас\s+)?(?:занимаются|делают|ведут|принимают)\b",
        x,
    ):
        return True

    if re.search(r"\bврачи\s+(?:занимаются|делают|ведут|принимают)\b", x):
        return True

    if re.search(r"\bспециалисты\s+(?:занимаются|делают|ведут|принимают)\b", x):
        return True

    if re.search(r"\bкто\s+принимает\b", x):
        return True

    if re.search(
        r"\b(?:расскаж\w*|покаж\w*)\s+(?:про|об)\s+.+?(?:\bврачи\b|\bврачей\b|\bврача\b|\bврач\b|\bдоктора\b|\bдоктор\b|\bспециалисты\b|\bспециалист\b)",
        x,
    ):
        return True

    if re.search(
        r"\b(?:стаж|опыт)\s+(?:у\s+)?(?:ваших|вашего|вашей|ваш\s+)?(?:врач|доктор)\w*",
        x,
    ):
        return True

    if re.search(
        r"\b(?:врачей|врача|врачи|доктора|доктор\w+)\s+(?:со\s+)?(?:стаж|опытом|опыт)\b",
        x,
    ):
        return True

    if re.search(r"\bсколько\b.+\b(?:врач|врачи|доктор|доктора|специалист|специалисты)", x):
        return True

    return False


def doctors_lookup(q: str, *, client_id: str) -> dict[str, Any] | None:
    """Результат для source_routing / app.

    routing:
      - doc — один врач, поле doc_id
      - cards — 2–3 врача, поле cards = list[DoctorListFact] для LLM-компоновки в app
      - overview — overview.md (в т.ч. 4+ совпадений или нет topic в каталоге)
    """
    q0 = (q or "").strip()
    if len(q0) < 2:
        return None

    q_lem = _lemma_set(q0)
    paths = _iter_doctor_paths()
    if not paths:
        return None

    # --- 1) Имя врача (name_short, name_full, aliases, H1); без doctor_intent
    best: tuple[float, DoctorPublic] | None = None
    for path in paths:
        fm, body, stem = _read_md_split(path)
        if fm.get("active") is False:
            continue
        rec = _doctor_record(path, fm, body, stem)
        local = 0.0
        for phrase in _name_candidates_from_fm(fm, _h1(body)):
            if not phrase:
                continue
            local = max(local, _name_hit_score(q_lem, phrase))
        if local >= 0.88:
            if best is None or local > best[0]:
                best = (local, rec)

    if best is not None:
        d = best[1]
        return {
            "routing": "doc",
            "doc_id": d["doc_id"],
            "doctor_name": d.get("name_full") or d.get("name_short") or d["doc_id"],
            "specialty": None,
            "cards": [d],
        }

    intent = doctor_intent_probe(q0)
    if not intent:
        return None

    # --- 2) Услуга из каталога (уверенный матч)
    cat_match = match_service_from_catalog(q0, client_id=client_id)
    if cat_match.get("matched_service_id") and bool(cat_match.get("is_confident")):
        sid = str(cat_match.get("matched_service_id") or "")
        by_svc = find_doctors_by_service(sid, client_id=client_id)
        by_svc = _dedupe_doctor_facts(by_svc)
        n = len(by_svc)
        if n == 1:
            d0 = by_svc[0]
            did = str(d0.get("doc_id") or "")
            return {
                "routing": "doc",
                "doc_id": did,
                "doctor_name": d0.get("name_full") or did,
                "specialty": None,
                "matched_service_id": sid,
                "cards": [d0],
            }
        if 2 <= n <= 3:
            return {
                "routing": "cards",
                "doc_id": None,
                "doctor_name": "Несколько врачей",
                "specialty": None,
                "matched_service_id": sid,
                "cards": by_svc,
            }
        if n >= 4:
            return {
                "routing": "overview",
                "doc_id": _OVERVIEW_ID,
                "doctor_name": "Наши врачи",
                "specialty": None,
                "matched_service_id": sid,
                "matching_doctors_total": n,
            }

    # --- 3) Ключевые слова специализации
    st = _specialty_topic_from_query(q0)
    if st is not _NO_SPECIALTY_KEYWORD:
        if st is None:
            return {
                "routing": "overview",
                "doc_id": _OVERVIEW_ID,
                "doctor_name": "Наши врачи",
                "specialty": None,
                "matching_doctors_total": None,
            }
        by_top = find_doctors_by_topic(str(st), client_id=client_id)
        by_top = _dedupe_doctor_facts(by_top)
        if not by_top:
            return {
                "routing": "overview",
                "doc_id": _OVERVIEW_ID,
                "doctor_name": "Наши врачи",
                "specialty": str(st),
                "matching_doctors_total": 0,
            }
        n = len(by_top)
        if n == 1:
            d0 = by_top[0]
            did = str(d0.get("doc_id") or "")
            return {
                "routing": "doc",
                "doc_id": did,
                "doctor_name": d0.get("name_full") or did,
                "specialty": str(st),
                "cards": [d0],
            }
        if 2 <= n <= 3:
            return {
                "routing": "cards",
                "doc_id": None,
                "doctor_name": "Несколько врачей",
                "specialty": str(st),
                "cards": by_top,
            }
        return {
            "routing": "overview",
            "doc_id": _OVERVIEW_ID,
            "doctor_name": "Наши врачи",
            "specialty": str(st),
            "matching_doctors_total": n,
        }

    # --- 4) Общий запрос про врачей
    if _is_generic_team_question(q0):
        return {
            "routing": "overview",
            "doc_id": _OVERVIEW_ID,
            "doctor_name": "Наши врачи",
            "specialty": None,
        }

    # Прежний эвристический «кто имплантолог» без явного слова из SPECIALTY_KEYWORDS
    if _is_staff_implant_question(q0, q_lem):
        by_top = find_doctors_by_topic("implantation", client_id=client_id)
        by_top = _dedupe_doctor_facts(by_top)
        n = len(by_top)
        if n == 1:
            d0 = by_top[0]
            did = str(d0.get("doc_id") or "")
            return {
                "routing": "doc",
                "doc_id": did,
                "doctor_name": d0.get("name_full") or did,
                "specialty": "implantation",
                "cards": [d0],
            }
        if 2 <= n <= 3:
            return {
                "routing": "cards",
                "doc_id": None,
                "doctor_name": "Несколько врачей",
                "specialty": "implantation",
                "cards": by_top,
            }
        if n >= 4:
            return {
                "routing": "overview",
                "doc_id": _OVERVIEW_ID,
                "doctor_name": "Наши врачи",
                "specialty": "implantation",
                "matching_doctors_total": n,
            }

    return None


def _is_staff_implant_question(q_raw: str, q_lem: set[str]) -> bool:
    low = q_raw.lower()
    if "врач" not in low and "доктор" not in low:
        return False
    if not (q_lem & frozenset({"имплант", "имплантация", "имплантолог"})):
        return False
    return bool(re.search(r"\b(кто|какой|какие|чей)\b", low, flags=re.I))


def doctor_name_probe(q: str, *, client_id: str | None = None) -> bool:
    """Только совпадение с именем/алиасом из md (кэш) или узкий фамильный suffix; без «врач + любая буква»."""
    x = _norm_query(q)
    if not x:
        return False

    phrases = cached_doctor_name_substrings(client_id=client_id)
    for phrase in sorted(phrases, key=len, reverse=True):
        if len(phrase) < 3:
            continue
        if phrase in x:
            return True

    # Нельзя использовать отдельный суффикс «ко» — он ловит бытовые слова («сколько»).
    if re.search(
        r"\b[а-яё]{3,}(?:ович|евич|ская|кой|ук|ова|ева|ина|ёва)\b",
        x,
        flags=re.I,
    ):
        return True
    return False
