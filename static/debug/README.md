# Dev-only: отладка `/ask` и retrieval

Папку **`static/debug/`** можно удалить целиком — на прод-виджет и на бэкенд это не влияет.

- **`ask-inspector.html`** — сырой JSON ответа, кандидаты `GET /__debug/retrieval` (нужен `X-Debug-Token` в локали, см. `.env` `DEBUG_TOKEN`).

Открыть: `http://127.0.0.1:9000/static/debug/ask-inspector.html` (порт как у Flask).
