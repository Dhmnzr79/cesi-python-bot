"""Жёсткий предфильтр строки «как к вам обращаться» — только явный мусор."""

import re

from session import normalize_phone

_EMAIL_RX = re.compile(r"\S+@\S+\.\S+", re.I)
_URL_RX = re.compile(r"(https?://|www\.)", re.I)
_MULTI_SPACE = re.compile(r"\s+")

# Фразы, которые почти наверняка не имя (одна подстрока в нижнем регистре).
_REJECT_SUBSTRINGS = (
    "болит зуб",
    "у меня бол",
    "у меня болит",
    "хочу запис",
    "записаться на",
    "оставить заяв",
    "оформить заяв",
    "есть ли парков",
    "сколько стоит",
    "как добраться",
    "где находит",
    "адрес клиник",
    "номер телефон",
    "телефон клиник",
    "можно ли запис",
    "когда вы работ",
    "график работ",
    "стоимость леч",
    "сколько будет",
    "расскажите про",
    "подскажите про",
)

# Отдельные токены-редфлаги (короткие вопросы/интенты, не ФИО).
_REJECT_TOKENS = frozenset(
    {
        "парковка",
        "парковку",
        "цена",
        "цены",
        "стоимость",
        "адрес",
        "телефон",
        "график",
        "запись",
        "записаться",
        "консультация",
        "консультацию",
        "имплант",
        "имплантация",
        "удалить",
        "удаление",
        "виниры",
        "брекеты",
    }
)

# Минимальный список грубой лексики (корни); без попытки покрыть весь интернет.
_PROFANITY = frozenset(
    {
        "хуй",
        "хуе",
        "хуя",
        "пизд",
        "ебан",
        "ебат",
        "ёбан",
        "бля",
        "сука",
        "мудак",
        "дурак",
        "дура",
        "придурок",
        "придурочн",
        "говно",
    }
)


def hard_reject_lead_name(text: str) -> bool:
    """True — строку не рассматриваем как кандидат в имя (явный мусор)."""
    s = (text or "").strip()
    if not s:
        return True
    if len(s) > 120:
        return True
    words = _MULTI_SPACE.split(s)
    if len(words) > 3:
        return True
    low = s.lower().replace("ё", "е")
    if any(ch.isdigit() for ch in s):
        return True
    if normalize_phone(s):
        return True
    if _EMAIL_RX.search(s):
        return True
    if _URL_RX.search(s):
        return True
    if "?" in s:
        return True
    for needle in _REJECT_SUBSTRINGS:
        if needle in low:
            return True
    for w in words:
        wl = w.lower().replace("ё", "е").strip(".,!?-—")
        if wl in _REJECT_TOKENS:
            return True
    for bad in _PROFANITY:
        if bad in low:
            return True
    return False
