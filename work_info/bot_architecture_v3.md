# Bot 3.0 — Архитектурный вектор для Cursor

## 1. Текущее состояние

### Что уже работает
- **Контентный слой** — markdown + YAML frontmatter с aliases, suggest_h3, suggest_refs, empathy-флагами, CTA. Сильная основа: знания и метаданные темы лежат рядом.
- **Индексация** — `build_index.py` режет по H2/H3, добавляет aliases в embedding-текст, нормализует векторы.
- **Диалоговый backend** — `app.py` умеет retrieval, LLM-rerank, ref-роутинг, память сессии, эмпатию, `/ask`, `/lead`, debug.
- **Логи** — `logging_setup.py` пишет JSONL — фундамент будущего дашборда.

### Что дал Flowise
Не финальная платформа, а этап для выработки спецификации: followup, handoff, видео, сценарий «ситуация», лимиты UI, порядок элементов. Это ценный опыт и источник правил.

---

## 2. Критичные проблемы

| Проблема | Суть |
|---|---|
| `app.py` — god file | Flask, retrieval, память, эмпатия, сборка ответа, лиды — всё в одном. Плюс блок сборки ответа продублирован ×4. |
| Нет policy-layer | Нет детерминированного слоя для: трактовки «да», показа CTA/видео/ситуации, мостиков между темами, handoff. |
| Состояние раздвоено | `SESSION_STATE` (эмпатия) и `SESS` (история) — два независимых словаря. Надо объединить. |
| Broad query → плохой чанк | Общий вопрос может попасть в узкий H3 вместо overview. Нужен механизм предпочтения overview. |
| Нет low-score защиты | При score < 0.40 бот всё равно отвечает из нерелевантного чанка. |

---

## 3. Сильные места — сохранить как фундамент

- **YAML как источник правды** — развить: добавить `topic`, `subtopic`, `video_key`, `next_buttons`, `handoff_policy`, `situation_allowed`.
- **Ref-роутинг кнопок** — кнопка несёт `label` + `ref`, не просто текст. Убирает класс ошибок retrieval.
- **Aliases в embedding** — пользователь спрашивает иначе чем написано в базе. Aliases решают этот разрыв.
- **Детерминированный роутинг до LLM** — contacts/prices через regex уже есть. Расширить на все очевидные случаи.

---

## 4. Целевая архитектура 3.0

```
┌─────────────────────────────────────────┐
│  Content layer                          │
│  markdown + YAML (знания + переходы)    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Retrieval layer                        │
│  candidates → broad query → prefer      │
│  overview → grouping → rerank → score   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Dialog state layer (один владелец)     │
│  history · turn_count · leadIntent      │
│  last_topic · shown_cta_topics          │
│  last_bot_action · last_offer_type      │
│  last_presented_buttons                 │
│  situation_pending · profile            │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Policy layer  ← ГЛАВНЫЙ СЛОЙ           │
│  детерминированные правила (не LLM)     │
│  что показывать · когда CTA · handoff   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Generation layer (LLM)                 │
│  текст · эмпатия · уточнение при        │
│  реальной двусмысленности после policy  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Response builder                       │
│  answer · followups · cta · video       │
│  situation · meta · debug               │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Integrations layer                     │
│  lead → n8n webhook → CRM / Telegram    │
└─────────────────────────────────────────┘
```

---

## 5. Policy layer — slot-policy

Вместо жёсткой “таблицы if-правил” policy использует **slot-модель**, описанную детально в `work_info/scenario_2.md`:

- YAML определяет, какие элементы вообще допустимы в теме:
  - `suggest_h3` → followup;
  - `video_key` → видео;
  - `situation_allowed` → кнопка «Рассказать о ситуации»;
  - `suggest_refs` → `suggest_ref` (не больше одного per doc).
- Экран имеет:
  - 1 слот под CTA;
  - 2 слота под дополнительные элементы.
- Дополнительные элементы заполняются по приоритету:
  - `followup → video → situation → suggest_ref`.

Спецрежимы имеют приоритет над slot-policy и временно отключают её:

- `lead_flow` — показываются только шаги воронки (имя, телефон, финальный шаг);
- `situation_pending` — показывается только экран ввода ситуации и кнопка «Назад к диалогу».

Policy не спорит с YAML, а только:

- исключает активные спецрежимы;
- убирает уже показанное (`video_shown`, `situation_offered`, покрытые followup и использованный `suggest_ref`);
- раскладывает допустимые элементы по двум слотам в порядке приоритета.

---

## 6. Обработка «да»

«Да» не имеет смысла само по себе — смысл берётся из последнего шага бота.

| Последний шаг бота | Трактовка «да» |
|---|---|
| Воронка записи | Продолжить воронку |
| Предложена подтема | Развернуть подтему |
| Предложен CTA | Запустить lead flow |
| Предложена «ситуация» | Включить situation_pending |
| Несколько равновероятных | Один уточняющий вопрос |
| Нет опоры | Дефолт: короткий ответ + одна кнопка |

Это не магия промпта — это явная логика в policy layer.

---

## 7. Кнопки и переходы

Кнопка всегда несёт два поля:
```json
{ "label": "Что входит в стоимость", "ref": "implants-pricing.md#cost-breakdown" }
```
Текст кнопки и маршрут — не одно и то же.

**Лимит на экране:** 1–2 followup + максимум один «тяжёлый» элемент (видео **или** ситуация **или** CTA). Не всё сразу.

---

## 8. Видео и сценарий «ситуация»

**Видео:**
- `video_key` в YAML темы → отдельный каталог `video_map.json` (title + url)
- Policy решает показывать или нет
- Не показывать если 2+ followup или идёт воронка

**Ситуация:**
- Отдельный сценарий, не просто кнопка
- `situation_pending` отключает retrieval на этот ход
- Note сохраняется и уходит в лид/CRM
- Повторно в той же теме не предлагать

---

## 9. API-контракт `/ask`

**Запрос:**
```json
{
  "q": "больно ли ставить имплант",
  "ref": "faq-pain.md#overview",
  "sid": "uuid",
  "client_id": "clinic_cesi"
}
```

**Ответ:**
```json
{
  "answer": "...",
  "followups": [
    { "label": "Какую анестезию используют", "ref": "faq-pain.md#anesthesia" }
  ],
  "cta": { "text": "Записаться на консультацию", "action": "booking" },
  "video": { "title": "...", "url": "..." },
  "situation": { "show": true },
  "meta": {
    "file": "faq-pain.md",
    "h2_id": "overview",
    "score": 0.87,
    "turn_count": 2,
    "client_id": "clinic_cesi",
    "sid": "uuid"
  }
}
```

---

## 10. Файловая структура после рефакторинга

```
bot/
├── app.py              # только Flask-роуты (/ask, /lead, /debug)
├── retriever.py        # embed, retrieve, rerank, broad_query_detect
├── llm.py              # промпты, эмпатия, вызов OpenAI
├── session.py          # единый session state (объединить SESS + SESSION_STATE)
├── policy.py           # 10 правил, логика кнопок, CTA, видео, ситуация
├── ux_builder.py       # build_response() — одна функция вместо ×4
├── lead_service.py     # валидация лида, сохранение, webhook → n8n, CRM-маршрутизация
├── config.py           # client_id, пути, модели, константы
├── meta_loader.py      # без изменений
├── logging_setup.py    # без изменений
├── build_index.py      # без изменений
│
├── md/
│   ├── clinic_cesi/
│   └── clinic_stoma_msk/
│
├── data/
│   ├── clinic_cesi/
│   │   ├── embeddings.npy
│   │   └── corpus.jsonl
│   └── clinic_stoma_msk/
│
└── clients/
    └── clients.json    # конфиг клиентов: name, cta_phone, tg_notify
```

---

## 11. Порядок работы в Cursor

### Этап 1 — Рефакторинг (не добавлять новые фичи)
1. Разнести `app.py` по модулям согласно структуре выше
2. Объединить `SESSION_STATE` и `SESS` в один `session.py`
3. Схлопнуть четыре одинаковых блока сборки ответа в `build_response()`
4. Добавить `client_id` во все слои

### Этап 2 — Ядро
5. Вынести policy layer в `policy.py` с таблицей правил
6. Добавить `last_bot_action`, `last_offer_type`, `last_presented_buttons` в `session.py`
7. Добавить broad query detector и low-score fallback в `retriever.py`
8. Перейти на SQLite вместо `SESS = {}` и JSONL-файла
9. Добавить structured outputs (JSON mode) в вызов OpenAI

### Этап 3 — Фичи
10. Нормализовать YAML-структуру под новые поля (video_key, next_buttons, situation_allowed)
11. Подключить видео и ситуацию как часть core
12. Простая страница `/admin` — дашборд из SQLite
13. Мультитенантность: изолированные индексы и сессии по `client_id`

---

## 12. Что не строить

- Multi-agent архитектура — избыточно для одного бота
- Векторная БД (Qdrant) — не нужна до 50k+ чанков
- BM25 / hybrid search — база написана простым языком, aliases закрывают задачу
- Fine-tuning — RAG меняет знания, fine-tuning меняет поведение; нужны знания
- LangChain / LangGraph — добавят абстракции поверх того что написано чище вручную

---

## 13. Source of truth — кто за что отвечает

Явная фиксация предотвращает дрейф в Cursor когда логику начинают класть куда попало.

| Что | Где живёт |
|---|---|
| Знания и маршруты переходов | YAML + markdown (`md/`) |
| Диалоговые решения (что показать, когда CTA) | `policy.py` |
| Текст ответа пользователю | `llm.py` (generation layer) |
| Состояние сессии | `session.py` |
| Конфиг клиента | `clients/clients.json` |
| Интеграции после лида | `lead_service.py` → n8n |

**Запрещённые паттерны:**
- "давай часть решим промптом" — если это бизнес-правило, оно идёт в `policy.py`
- "давай положим в meta и фронт разберётся" — фронт не принимает решений
- "давай захардкодим в app.py" — app.py только роуты

---

## 14. last_bot_action — обязательное поле состояния

Без этого поля policy не может корректно трактовать короткие ответы («да», «угу», «давайте», «хочу»).

**Что хранить в session state:**
```python
session = {
    # ... остальные поля ...
    "last_bot_action": "offered_subtopic",   # что предложил бот последним
    "last_offer_type": "followup",           # тип предложения
    "last_presented_buttons": [              # какие кнопки были на экране
        { "label": "Этапы имплантации", "ref": "faq-stages.md#overview" }
    ]
}
```

**Допустимые значения `last_bot_action`:**
```
offered_subtopic      → развернуть предложенную подтему
offered_cta           → запустить lead flow
offered_situation     → включить situation_pending
in_lead_flow          → продолжить воронку
offered_handoff       → подтвердить передачу
none                  → дефолт: короткий ответ + одна кнопка
```

**Правило:** policy читает `last_bot_action` первым делом при любом коротком ответе. LLM подключается только если после этого всё ещё есть двусмысленность.

---

## 15. Broad query → prefer overview

Одна из центральных проблем по тестам: общий вопрос попадает в узкий H3 вместо overview.

**Принцип:**
- Если вопрос общий (короткий, без конкретики) и документ содержит overview + H3 → предпочитать overview
- H3 использовать как followup-углубление, не как главный ответ на общий вопрос
- Если топ-3 кандидата из одного файла → брать overview этого файла

**Реализация в `retriever.py`:**
```python
def broad_query_detect(q: str) -> bool:
    # короткий вопрос без конкретных терминов = broad
    return len(q.split()) <= 5 and not any(
        term in q.lower() for term in ["цена", "стоимость", "адрес", "телефон"]
    )

def prefer_overview(candidates: list) -> dict:
    # если broad query и есть overview в топ-3 — взять его
    for c in candidates[:3]:
        if _is_overview_by_ids(c.get("h2_id"), c.get("h3_id")):
            return c
    return candidates[0]
```

**Low-score защита:**
```python
LOW_SCORE_THRESHOLD = 0.40

if top_score < LOW_SCORE_THRESHOLD:
    return fallback_response(
        answer="Не нашёл точного ответа на этот вопрос. Уточните или выберите тему.",
        cta=default_cta(client_id)
    )
```

---

## 16. Безопасность сервера

### Обязательный минимум

**nginx:**
```nginx
# Rate limiting — не более 10 запросов в минуту с одного IP
limit_req_zone $binary_remote_addr zone=ask:10m rate=10r/m;

location /ask {
    limit_req zone=ask burst=5 nodelay;
    proxy_pass http://127.0.0.1:8000;
}

# CORS — только доверенные домены клиентов
add_header Access-Control-Allow-Origin "https://клиент.ru" always;
```

**Firewall (ufw):**
```bash
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 22/tcp   # или нестандартный порт SSH
ufw deny все остальное
```

**Приложение:**
- Все секреты только в `.env`, никогда в коде
- Gunicorn в проде, не Flask dev server: `gunicorn -w 2 -b 127.0.0.1:8000 app:app`
- `/admin` и `/debug` — под basic auth или отдельным токеном
- Логи не должны содержать OpenAI API key, телефоны, персональные данные

### Валидация client_id

`client_id` приходит от фронта — доверять нельзя без проверки.

```python
# config.py
ALLOWED_CLIENTS = set(clients_json.keys())  # {"clinic_cesi", "clinic_stoma_msk"}

# app.py — в начале каждого /ask
client_id = data.get("client_id", "").strip()
if client_id not in ALLOWED_CLIENTS:
    return jsonify({"error": "unknown_client"}), 403
```

Без этой проверки любой может запросить данные чужого клиента или сломать индекс.

### Бэкапы

Три вещи которые нужно бэкапить ежедневно:

```bash
# cron: 0 3 * * * /srv/backup.sh
rsync -av /srv/bot/md/        backup:/bot-backup/md/
rsync -av /srv/bot/data/      backup:/bot-backup/data/
rsync -av /srv/bot/bot.db     backup:/bot-backup/db/
```

- `md/` — контент клиентов (долго восстанавливать вручную)
- `data/` — эмбеддинги (пересчитываются, но это время и деньги на API)
- `bot.db` — SQLite с сессиями и логами (история диалогов, аналитика)
