"""Промпты и вызовы OpenAI (чат); эмпатия."""
import json

from openai import OpenAI

from config import (
    CHAT_JSON_MODE,
    CHAT_MODEL,
    EMPATHY_ON,
    MEMORY_ON,
    OPENAI_API_KEY,
    TRIGGERS_COMPILED,
)
from session import (
    is_first_in_topic,
    mem_add_bot,
    mem_add_user,
    mem_context,
    update_topic_empathy,
)

client = OpenAI(api_key=OPENAI_API_KEY)

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
