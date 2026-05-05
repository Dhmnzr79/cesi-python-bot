"""Константы, пути, модели, regex. Секреты только из окружения."""
import os
import re

from dotenv import load_dotenv

load_dotenv()

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMB_MODEL = os.getenv("MODEL_EMBED", "text-embedding-3-large")
CHAT_MODEL = os.getenv("MODEL_CHAT", "gpt-5.4-mini")
QUERY_REWRITE_MODEL = (os.getenv("MODEL_QUERY_REWRITE") or "").strip() or "gpt-5.4-nano"
RERANK_MODEL = (os.getenv("MODEL_RERANK") or "").strip() or "gpt-5.4-mini"
LEAD_NAME_CLASSIFY_MODEL = (os.getenv("MODEL_LEAD_NAME") or "").strip() or CHAT_MODEL

# --- Намерение «записаться» (regex + при необходимости LLM) ---
BOOKING_INTENT_LLM_ON = os.getenv("BOOKING_INTENT_LLM_ON", "1").lower() in (
    "1",
    "true",
    "yes",
)
BOOKING_INTENT_LLM_MODEL = (os.getenv("BOOKING_INTENT_LLM_MODEL") or "").strip() or "gpt-5.4-nano"
PRICE_INTENT_LLM_ON = os.getenv("PRICE_INTENT_LLM_ON", "1").lower() in (
    "1",
    "true",
    "yes",
)
PRICE_INTENT_LLM_MODEL = (os.getenv("PRICE_INTENT_LLM_MODEL") or "").strip() or CHAT_MODEL
SAFETY_CLASSIFY_MODEL = (os.getenv("MODEL_SAFETY_CLASSIFY") or "").strip() or "gpt-5.4-nano"
SAFETY_RED_CONFIDENCE_THRESHOLD = float(os.getenv("SAFETY_RED_CONFIDENCE_THRESHOLD", "0.8"))
COMPLAINT_CLASSIFY_MODEL = (os.getenv("MODEL_COMPLAINT_CLASSIFY") or "").strip() or "gpt-5.4-nano"
QUERY_REWRITE_ON = os.getenv("QUERY_REWRITE_ON", "1").lower() in ("1", "true", "yes")
QUERY_REWRITE_MAX_MESSAGES = int(os.getenv("QUERY_REWRITE_MAX_MESSAGES", "10"))
# Подстроки в ответе rewrite → отбросить (утечка инструкции / мусор). Разделитель |
_rewrite_reject_raw = os.getenv(
    "REWRITE_REJECT_SUBSTRINGS",
    "врач, процедура, симптом, зуб, материал|ключевые сущности",
)
REWRITE_REJECT_SUBSTRINGS: tuple[str, ...] = tuple(
    x.strip().lower() for x in _rewrite_reject_raw.split("|") if x.strip()
)
QUERY_REWRITE_VALIDATE_OVERLAP = os.getenv("QUERY_REWRITE_VALIDATE_OVERLAP", "1").lower() in (
    "1",
    "true",
    "yes",
)

# --- HTTP / app ---
PORT = int(os.getenv("PORT", "9000"))
DEBUG_TOKEN = os.getenv("DEBUG_TOKEN", "dev-debug")
INPUT_MAX_CHARS = int(os.getenv("INPUT_MAX_CHARS", "600"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
RATE_LIMIT_MAX_PER_IP = int(os.getenv("RATE_LIMIT_MAX_PER_IP", "40"))
ANTI_SPAM_NO_INTENT_TURNS = int(os.getenv("ANTI_SPAM_NO_INTENT_TURNS", "10"))
ANTI_SPAM_BURST_WINDOW_SEC = int(os.getenv("ANTI_SPAM_BURST_WINDOW_SEC", "120"))
ANTI_SPAM_BURST_MESSAGES = int(os.getenv("ANTI_SPAM_BURST_MESSAGES", "6"))

# --- Paths ---
DATA_DIR = os.getenv("DATA_DIR", "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
EMB_PATH = os.path.join(DATA_DIR, "embeddings.npy")
SQLITE_PATH = os.getenv("SQLITE_PATH", os.path.join(DATA_DIR, "bot.db"))

# --- Retrieval / policy пороги ---
LOW_SCORE_THRESHOLD = float(os.getenv("LOW_SCORE_THRESHOLD", "0.33"))
BROAD_QUERY_MAX_WORDS = int(os.getenv("BROAD_QUERY_MAX_WORDS", "5"))

# Алиас по корпусу: «сильный» — как раньше 0.82; «мягкий» — подстраховка у LOW_SCORE (не второй порог на клиента).
ALIAS_STRONG_THRESHOLD = float(os.getenv("ALIAS_STRONG_THRESHOLD", "0.82"))
ALIAS_SOFT_THRESHOLD = float(os.getenv("ALIAS_SOFT_THRESHOLD", "0.72"))


# --- Ответ при низком score ---
DEFAULT_CTA_TEXT = os.getenv("DEFAULT_CTA_TEXT", "Записаться на консультацию")
DEFAULT_CTA_ACTION = os.getenv("DEFAULT_CTA_ACTION", "lead")

# --- LLM: JSON-ответ { "answer": "..." } ---
CHAT_JSON_MODE = os.getenv("CHAT_JSON_MODE", "1").lower() in ("1", "true", "yes")

# --- Явное намерение записаться (обход запрета CTA при turn_count < 2) ---
# Не матчим голые «консультац» / «приём» — иначе ловятся контентные вопросы.
# «записаться» не после как/где/куда (FAQ «как записаться»).
BOOKING_INTENT_RE = re.compile(
    r"(?:"
    r"запишите\s+меня"
    r"|хочу\s+запис(аться|ать)\b"
    r"|запись\s+на\s+(?:консультац|приём|прием)"
    r"|остав(ить|лю)\s+заявку"
    r"|(?<!\bкак\s)(?<!\bгде\s)(?<!\bкуда\s)\bзапис(аться|ать)\b"
    r"(?:\s+на\s+(?:консультац|приём|прием))?"
    r")",
    re.I | re.U,
)

# --- Multi-tenant (сейчас один клиент; неизвестный id → 403) ---
DEFAULT_CLIENT_ID = os.getenv("DEFAULT_CLIENT_ID", "default").strip() or "default"
_ac_raw = os.getenv("ALLOWED_CLIENTS", "").strip()
if _ac_raw:
    ALLOWED_CLIENTS = frozenset(x.strip() for x in _ac_raw.split(",") if x.strip())
else:
    ALLOWED_CLIENTS = frozenset({DEFAULT_CLIENT_ID})

# --- Детерминированный роутинг до LLM ---
CONTACTS_RE = re.compile(
    r"(адрес|где.*находитесь|как\s+(доехать|проехать)|время\s+работы|график|телефон|whatsapp|карта|расположение)",
    re.I,
)
PRICES_RE = re.compile(
    r"(цена|стоимост|сколько\s+стоит|прайс|расценк|по\s+цене|сколько\s+будет|сколько\s+руб)",
    re.I,
)
PRICE_LOOKUP_RE = re.compile(
    r"(цена|стоимост|сколько\s+стоит|прайс|расценк|по\s+цене|сколько\s+будет|сколько\s+руб|сколько\s+обойд[её]тся?)",
    re.I,
)
# Без «скидк/рассрочк»: вопросы про скидки, полис, рассрочку — обычный retrieval (payment_terms и т.д.),
# а не price_concern к конкретной услуге.
PRICE_CONCERN_RE = re.compile(
    r"(дорог|почему\s+так\s+дорого|слишком\s+дорого|высокая\s+цена|не\s+потяну|не\s+по\s+карману|дешевле|снизить\s+стоимост)",
    re.I,
)

PRICE_SERVICE_MATCH_STRONG = float(os.getenv("PRICE_SERVICE_MATCH_STRONG", "0.62"))

# --- Память диалога ---
MEMORY_ON = True
MAX_TURNS = 8
MAX_IDLE_SEC = 60 * 60

# --- Кэш retrieval ---
RETRIEVE_CACHE_TTL_SEC = int(os.getenv("RETRIEVE_CACHE_TTL_SEC", "120"))
RETRIEVE_CACHE_MAXSIZE = int(os.getenv("RETRIEVE_CACHE_MAXSIZE", "512"))

# --- Эмпатия ---
EMPATHY_ON = True
TRIGGERS = {
    "fear_pain": r"(боюс|страшн|тревог|паник|боль|болит|болезнен|анестез|заморозк|укол)",
    "safety": r"(опасн|зараж|инфекц|стерил|безопасн|чистот|противопоказан|риск)",
    "price": r"(дорог|дешев|стоимост|цена|сколько стоит|рассрочк)",
    "timing": r"(сколько времен|как долго|срок|долго|за один день|быстрее)",
    "indications": r"(подходит ли|можно ли мне|мой случай|показан|показания)",
    "support": r"(пережив|сомнева|не уверен|не уверена|тяну ли|поможете|помогите)",
}
TRIGGERS_COMPILED = {k: re.compile(v, re.I | re.U) for k, v in TRIGGERS.items()}

_LLM_PRICE_IN_PER_1M = float(os.getenv("BOT_LLM_USD_PER_1M_PROMPT", "0") or "0")
_LLM_PRICE_OUT_PER_1M = float(os.getenv("BOT_LLM_USD_PER_1M_COMPLETION", "0") or "0")


def estimate_llm_usage_usd(
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    """Грубая оценка затрат для дашборда. Нули env → вернуть None (не гадать)."""
    if _LLM_PRICE_IN_PER_1M <= 0 and _LLM_PRICE_OUT_PER_1M <= 0:
        return None
    pt = int(prompt_tokens or 0)
    ct = int(completion_tokens or 0)
    return round(
        (pt * _LLM_PRICE_IN_PER_1M + ct * _LLM_PRICE_OUT_PER_1M) / 1_000_000.0,
        8,
    )


if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")


def resolve_client_id(raw: str | None) -> str | None:
    cid = (raw or "").strip() or DEFAULT_CLIENT_ID
    return cid if cid in ALLOWED_CLIENTS else None


def default_cta_dict() -> dict:
    return {"text": DEFAULT_CTA_TEXT, "action": DEFAULT_CTA_ACTION}
