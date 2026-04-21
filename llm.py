"""Промпты и вызовы OpenAI (чат); эмпатия."""
import json
import re

from openai import OpenAI

from config import (
    CHAT_JSON_MODE,
    CHAT_MODEL,
    EMPATHY_ON,
    MEMORY_ON,
    OPENAI_API_KEY,
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
    mem_add_bot,
    mem_add_user,
    mem_context,
    mem_get,
    update_topic_empathy,
)

client = OpenAI(api_key=OPENAI_API_KEY)
logger = get_logger("bot")

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
            max_tokens=200,
            response_format={"type": "json_object"},
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
            query_raw=q0[:200],
            err=str(e)[:300],
        )
        return q0


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
    mem_add_user(session_id, user_q)
    mem_txt, profile = mem_context(session_id)

    messages, use_empathy, doc_key = build_messages_for_gpt(
        user_q, context_md, meta, session_id
    )

    if mem_txt and MEMORY_ON:
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] = f"Недавний диалог:\n{mem_txt}\n\n" + msg["content"]
                break

    kwargs = dict(model=CHAT_MODEL, temperature=0.3, messages=messages)
    if CHAT_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}
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

    mem_add_bot(session_id, answer)
    update_topic_empathy(session_id, doc_key, use_empathy)

    return answer, profile
