Инструкция для Cursor — ценовой слой бота

Контекст
Бот на Python для стоматологических клиник. Уже есть: md-база с YAML, retrieval, policy layer, session state. Добавляем ценовой слой поверх существующей архитектуры.

Файловая структура
clients/
  clinic_cesi/
    service_catalog.json
    prices.json
  clinic_stoma_msk/
    service_catalog.json
    prices.json

md/
  clinic_cesi/
    implantation__pricing__implants.md  ← ценовой нарратив (сложные услуги)
    implantation__faq__cost.md          ← возражения по цене, БЕЗ цифр
    implantation__service__all_on_4.md  ← услуга, БЕЗ цифр
    ...остальные md без цифр...

1. prices.json
Единственное место где живут цифры. Три типа цен:
json{
  "wisdom_tooth_simple": {
    "name": "Удаление зуба мудрости",
    "price_type": "range",
    "value_min": 4500,
    "value_max": 9000,
    "currency": "RUB",
    "note": "сложность определяется на осмотре"
  },
  "pulpitis": {
    "name": "Лечение пульпита",
    "price_type": "from",
    "value": 6800,
    "currency": "RUB",
    "note": null
  },
  "implant_implantium": {
    "name": "Имплант Implantium под ключ",
    "price_type": "fixed",
    "value": 76200,
    "currency": "RUB",
    "note": null
  }
}
Рендеринг цены — только кодом, никогда не LLM:
pythondef format_price(p: dict) -> str:
    if p["price_type"] == "fixed":
        return f"{p['value']:,} ₽".replace(",", " ")
    if p["price_type"] == "from":
        return f"от {p['value']:,} ₽".replace(",", " ")
    if p["price_type"] == "range":
        return f"{p['value_min']:,} – {p['value_max']:,} ₽".replace(",", " ")

2. service_catalog.json
Единый реестр всех услуг. Роутер и источник данных одновременно.
json{
  "wisdom_tooth_extraction": {
    "title": "Удаление зуба мудрости",
    "aliases": ["удаление зуба мудрости", "удалить восьмёрку", "восьмёрка удаление"],
    "response_mode": "card",
    "active": true,
    "facts": [
      "проводится под анестезией",
      "подходит для простых и сложных случаев",
      "врач даёт рекомендации по восстановлению"
    ],
    "price_key": "wisdom_tooth_simple",
    "price_display": "always",
    "md_entry_ref": null,
    "price_ref": null,
    "suggest_refs": []
  },
  "implant_single": {
    "title": "Имплантация зуба",
    "aliases": ["имплантация зуба", "поставить имплант", "имплант цена"],
    "response_mode": "card",
    "active": true,
    "facts": [],
    "price_key": null,
    "price_display": "on_request",
    "md_entry_ref": "implantation__service__all_on_4",
    "price_ref": "implantation__pricing__implants",
    "suggest_refs": [
      {
        "label": "Цены на импланты",
        "ref": "implantation__pricing__implants.md#korotko"
      }
    ]
  }
}
Поля:
ПолеЗначенияСмыслmd_entry_refстрока / nullnull → facts из карточки, строка → идём в mdprice_refстрока / nullnull → prices.json, строка → идём в ценовой mdprice_displayalways / on_requestalways → цена в первом ответе, on_request → только при price_lookupsuggest_refsмассив / []максимум 1 элемент для card-ответаactivetrue / falsefalse → карточка отключена без удаления

3. Классификатор intent
Файл intent_classifier.py. Три уровня:
pythonimport re

PRICE_LOOKUP_PATTERNS = [
    r"сколько стоит", r"какая цена", r"цена на", r"стоимость",
    r"почём", r"за сколько", r"прайс", r"цены у вас",
]

PRICE_CONCERN_PATTERNS = [
    r"почему (так )?дорого", r"дешевле", r"дорогов?ато",
    r"не по карману", r"можно ли сэконом", r"за что такая цена",
    r"оправдан", r"стоит ли столько",
]

def classify_intent(q: str, client) -> str:
    q_lower = q.lower()

    # Уровень 1 — regex concern (бесплатно)
    if any(re.search(p, q_lower) for p in PRICE_CONCERN_PATTERNS):
        return "price_concern"

    # Уровень 2 — regex lookup (бесплатно)
    if any(re.search(p, q_lower) for p in PRICE_LOOKUP_PATTERNS):
        return "price_lookup"

    # Уровень 3 — LLM классификатор (только если regex промахнулся)
    return _llm_classify(q, client)

def _llm_classify(q: str, client) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1,
        messages=[{
            "role": "user",
            "content": f"""Сообщение пользователя в чате стоматологии: «{q}»

Определи тип. Ответь только цифрой:
1 — спрашивает конкретную цену или стоимость
2 — возражение или страх по поводу цены
3 — другое"""
        }]
    )
    result = response.choices[0].message.content.strip()
    return {"1": "price_lookup", "2": "price_concern"}.get(result, "other")

4. price_resolver.py
pythondef resolve(q: str, intent: str, client_id: str) -> RetrievalResult:
    catalog = load_catalog(client_id)
    prices = load_prices(client_id)

    match = fuzzy_match_catalog(q, catalog)

    # Услуга не найдена
    if not match:
        return fallback_response(client_id)

    # price_concern — всегда в faq__cost, не в прайс
    if intent == "price_concern":
        return retrieve_from_md("implantation__faq__cost", client_id)

    # price_lookup
    if intent == "price_lookup":
        if match.get("price_ref"):
            # Сложная цена — идём в ценовой md
            return retrieve_from_md(match["price_ref"], client_id)
        else:
            # Простая цена — из prices.json
            price = prices.get(match["price_key"])
            return PriceResult(
                service=match["title"],
                price_formatted=format_price(price) if price else None
            )

    # other — обычный роутинг
    if match.get("md_entry_ref"):
        return retrieve_from_md(match["md_entry_ref"], client_id)
    else:
        return retrieve_from_catalog(match, prices)

5. Policy для card-ответа
В policy.py добавить ветку. Для source = "catalog" slot-policy из scenario_2 не применяется:
pythonif result.source == "catalog":
    return ResponsePayload(
        answer=build_card_answer(result),
        followups=[],
        video=None,
        situation=None,
        suggest_ref=result.suggest_refs[0] if result.suggest_refs else None,
        cta=default_cta(client_id)
    )

6. Промпты
Card-ответ (хвостовая услуга):
pythonCARD_PROMPT = """
Ты — консультант стоматологической клиники.
Вопрос пользователя: «{question}»

Услуга: {title}
Факты: {facts}
Стоимость: {price_formatted}

Напиши живой ответ 2–3 предложения. Используй только факты из списка.
Строку стоимости вставь дословно — не перефразируй цифру.
Не предлагай записаться.
"""
Price_lookup простой (из prices.json):
pythonPRICE_SIMPLE_PROMPT = """
Ты — консультант стоматологической клиники.
Вопрос: «{question}»
Услуга: {title}
Стоимость: {price_formatted}

1–2 живые фразы. Цену вставь дословно. Не предлагай записаться.
"""

7. Итоговый поток
Запрос
  │
  ▼
classify_intent()
  │
  ├── price_concern → faq__cost.md → эмпатия + воронка
  │
  ├── price_lookup
  │     ├── price_ref есть → ценовой md → развёрнутый ответ
  │     └── price_ref null → prices.json → короткий ответ
  │
  └── other
        ├── md_entry_ref есть → md → полная воронка
        ├── md_entry_ref null → facts + price → card-ответ
        └── catalog не нашёл → vector_search → fallback

8. Правила контента — обязательно

Цифры цен только в prices.json и implantation__pricing__implants.md
Все остальные md-документы — без цифр
faq__cost.md — только эмпатия и аргументы, цифр нет
facts в карточках — тезисы, не готовые фразы
suggest_refs в карточке — максимум 1 элемент
Aliases пополнять из логов по мере накопления реальных запросов