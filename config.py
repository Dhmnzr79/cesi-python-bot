"""Константы, пути, модели, regex. Секреты только из окружения."""
import os
import re

from dotenv import load_dotenv

load_dotenv()

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMB_MODEL = os.getenv("MODEL_EMBED", "text-embedding-3-small")
CHAT_MODEL = os.getenv("MODEL_CHAT", "gpt-4o-mini")
QUERY_REWRITE_MODEL = (os.getenv("MODEL_QUERY_REWRITE") or "").strip() or CHAT_MODEL
QUERY_REWRITE_ON = os.getenv("QUERY_REWRITE_ON", "1").lower() in ("1", "true", "yes")
QUERY_REWRITE_MAX_MESSAGES = int(os.getenv("QUERY_REWRITE_MAX_MESSAGES", "10"))

# --- HTTP / app ---
PORT = int(os.getenv("PORT", "9000"))
DEBUG_TOKEN = os.getenv("DEBUG_TOKEN", "dev-debug")

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

# Selective rerank: узкое окно (как было) + «у порога» low_score при малом зазоре 1–2 места.
RERANK_TOP_MIN = float(os.getenv("RERANK_TOP_MIN", "0.20"))
RERANK_TOP_MAX = float(os.getenv("RERANK_TOP_MAX", "0.62"))
RERANK_GAP_MAX = float(os.getenv("RERANK_GAP_MAX", "0.05"))
RERANK_NEAR_LOW_TOP_MAX = float(os.getenv("RERANK_NEAR_LOW_TOP_MAX", "0.36"))
RERANK_NEAR_LOW_GAP_MAX = float(os.getenv("RERANK_NEAR_LOW_GAP_MAX", "0.10"))

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

# --- Память диалога ---
MEMORY_ON = True
MAX_TURNS = 8
MAX_IDLE_SEC = 60 * 60

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

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")


def resolve_client_id(raw: str | None) -> str | None:
    cid = (raw or "").strip() or DEFAULT_CLIENT_ID
    return cid if cid in ALLOWED_CLIENTS else None


def default_cta_dict() -> dict:
    return {"text": DEFAULT_CTA_TEXT, "action": DEFAULT_CTA_ACTION}
