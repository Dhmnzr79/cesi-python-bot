# Инвентаризация промптов, заготовок и инструкций

Документ для редактирования всех текстовых управляющих слоев бота: промптов LLM, системных инструкций, шаблонов ответов и сервисных текстов.

## 1) LLM-промпты и инструкции (файл `llm.py`)

### 1.1 Retrieval rewrite
- **Код:** `_REWRITE_SYSTEM`
- **Что это:** system prompt для переписывания запроса под retrieval.
- **Где используется:** `rewrite_query_for_retrieval()` -> OpenAI `chat.completions.create(...)`.
- **Текст:**
  - "Ты формулируешь поисковый запрос для семантического поиска по базе знаний стоматологии..."
  - "Ответь одним JSON-объектом с ключом `search_query`..."

### 1.2 Facts-card generation
- **Код:** `_FACTS_CARD_SYSTEM`
- **Что это:** system prompt для генерации ответа только по `facts` из `service_catalog`.
- **Где используется:** `generate_facts_card_answer()`.
- **Текст:**
  - "Ты помощник стоматологической клиники..."
  - "Ответь одним JSON-объектом с ключом `answer`."

### 1.3 Базовый промпт ответа по чанку
- **Код:** `BASE_SYSTEM`
- **Что это:** базовая роль и стиль ответа.
- **Где используется:** `build_messages_for_gpt()` -> формирует `system_prompt`.
- **Текст:** "Ты — спокойный и доброжелательный врач-имплантолог..."

### 1.4 Эмпатийная надстройка
- **Код:** `EMPATHY_ADDON`
- **Что это:** инструкция на добавление 1 короткой эмпатийной фразы.
- **Где используется:** `build_messages_for_gpt()` при `use_empathy=True`.

### 1.5 JSON-формат ответа
- **Код:** `JSON_ANSWER_RULE`
- **Что это:** требование строгого JSON.
- **Где используется:** `build_messages_for_gpt()` при `CHAT_JSON_MODE=True`.

### 1.6 User message template в `build_messages_for_gpt()`
- **Что это:** шаблон `user`-сообщения для генерации по чанку.
- **Где используется:** `build_messages_for_gpt()`.
- **User message template:**
  - "Вопрос пациента:\n{...}\n\nКонтент для ответа (markdown, цитируй по смыслу, не выдумывай):\n{...}"

### 1.7 Классификатор имени для lead-flow
- **Код:** `_NAME_CLASSIFY_SYSTEM`
- **Что это:** system prompt классификации имени (`valid_name|invalid_name|unsure`).
- **Где используется:** `classify_lead_name_shape()`.

### 1.8 Классификатор booking-intent
- **Код:** `_BOOKING_INTENT_SYSTEM`
- **Что это:** system prompt (`wants_booking: bool`).
- **Где используется:** `classify_booking_wants_appointment()`.

### 1.9 Классификатор ценового интента
- **Код:** `_PRICE_INTENT_SYSTEM`
- **Что это:** system prompt (`price_lookup|price_concern|other`).
- **Где используется:** `classify_price_intent()`.

### 1.10 Единый intent classifier
- **Код:** `_INTENT_CLASSIFY_SYSTEM`
- **Что это:** system prompt (`contacts|price_lookup|price_concern|content`).
- **Где используется:** `classify_intent()`, далее в `app.py` в `ask()`.

### 1.11 Fallback-текст LLM
- **Код:** `LLM_FALLBACK_ANSWER`
- **Что это:** дефолтный ответ при сбое/пустом ответе LLM.
- **Где используется:** `generate_answer_with_empathy()`.
- **Текст по умолчанию:** "Извините, сейчас есть техническая задержка..."

---

## 2) Текстовые заготовки lead/situation flow (файл `app.py`)

### 2.1 Словарь `TXT`
- **Что это:** централизованные реплики сценарного флоу.
- **Где используется:** передается в `handle_flows(..., txt=TXT)` из `ask()`.

Список ключей и назначение:
- `lead_name_prompt` — старт сбора имени.
- `lead_name_retry` — повторный запрос имени.
- `lead_name_hard` — hard reject для невалидного ввода.
- `lead_name_invalid` — soft reject для не-имени.
- `lead_name_confirm_tpl` — подтверждение неоднозначного имени.
- `lead_name_reenter` — повторный ввод имени после "нет".
- `lead_phone_prompt_tpl` — запрос телефона после имени.
- `lead_phone_retry` — повторный запрос телефона.
- `lead_submit_ok` — успешная отправка лида.
- `lead_submit_error` — ошибка отправки лида.
- `situation_prompt` — начало сбора "вашей ситуации".
- `situation_retry_short` — слишком короткое описание ситуации.
- `situation_to_lead_name` — переход от ситуации к имени.
- `situation_back_fallback` — fallback при возврате назад.
- `followup_choose_topic` — предложение выбрать тему при ambiguous yes.

---

## 3) Шаблоны и тексты JSON-ответов API (файл `ux_builder.py`)

### 3.1 Базовые response-функции
- `empty_question_response()`  
  - Текст: "Уточните вопрос."
- `no_candidates_response()`  
  - Текст: "Пока не нашёл подходящий материал в базе..."
- `reset_session_response()`  
  - Текст: "Начнём заново. Чем помочь?"
- `internal_error_response()`  
  - Текст: "Извините, не получилось ответить..."
- `low_score_response()`  
  - Текст: "Не нашёл точного ответа на этот вопрос..."
  - CTA берётся из `config.default_cta_dict()` (`DEFAULT_CTA_TEXT`, `DEFAULT_CTA_ACTION`).

### 3.2 Price layer payloads
- `build_price_lookup_payload()`  
  - Форматы:
    - `{title}: {price}.`
    - + "Важно: {note}."
    - fallback: "По услуге «...», сейчас не вижу точной цены..."
- `build_price_concern_payload()`  
  - Текст: "Понимаю сомнение по стоимости..."
- `build_price_clarify_payload()`  
  - Текст: "Уточните, пожалуйста, какую именно услугу..."

### 3.3 Catalog facts card
- `build_service_facts_card_payload()`
- Поведение:
  - сначала пытается `generate_facts_card_answer(...)` (LLM),
  - fallback без LLM: `"{title}\n\n• fact1\n• fact2..."`.

---

## 4) Сценарные кнопки/шаблоны в flow-логике (файл `flow_handlers.py`)

### 4.1 Quick replies подтверждения имени
- **Код:** `_name_confirm_quick_replies()`
- **Тексты кнопок:**
  - "Да"
  - "Нет, введу по-другому"
- **Где используется:** ветка `lead_intent == confirming_name`.

### 4.2 Использование `TXT` в flow
- `flow_handlers.py` сам тексты почти не хранит, а использует ключи из `TXT`.
- Ключевые точки:
  - start lead,
  - collecting name/phone,
  - situation start/retry/back,
  - followup disambiguation.

---

## 5) Текстовые дефолты в конфиге (файл `config.py`)

### 5.1 CTA дефолты
- `DEFAULT_CTA_TEXT = "Записаться на консультацию"`
- `DEFAULT_CTA_ACTION = "lead"`
- **Где используется:** `default_cta_dict()` -> `ux_builder.low_score_response()` и price concern CTA.

### 5.2 Фильтр rewrite-мусора
- `REWRITE_REJECT_SUBSTRINGS` default:
  - `"врач, процедура, симптом, зуб, материал|ключевые сущности"`
- **Где используется:** `llm.validated_retrieval_rewrite()`.

---

## 6) Текстовые шаблоны e-mail лида (файл `lead_service.py`)

- **Что это:** шаблон тела email-лида для администратора.
- **Где используется:** `_send_lead_email()`.
- **Структура строк:**
  - "Время (UTC): ..."
  - "Имя: ..."
  - "Телефон: ..."
  - "Ситуация: ..."
  - "SID: ..."
  - "Client ID: ..."
  - "Intent: ..."

---

## 7) Что редактируем обычно в первую очередь

Для тюнинга поведения бота чаще всего правятся:
1. `llm.py`: `BASE_SYSTEM`, `EMPATHY_ADDON`, `_INTENT_CLASSIFY_SYSTEM`, `_PRICE_INTENT_SYSTEM`
2. `app.py`: словарь `TXT`
3. `ux_builder.py`: тексты fallback/price templates
4. `config.py`: `DEFAULT_CTA_TEXT`, `REWRITE_REJECT_SUBSTRINGS`

---

## 8) Примечание по контенту MD

Отдельно от этого документа есть тексты базы знаний в `md/*.md` (контент для retrieval).  
Их можно редактировать независимо от промптов/шаблонов.

