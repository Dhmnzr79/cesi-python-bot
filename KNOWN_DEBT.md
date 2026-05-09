# Active Technical Debt v5

## Contacts intent post-Resolver overlay (PR #1.3)

В app.py есть тонкий overlay после Resolver: если CONTACTS_RE matched после Resolver → принудительно intent='contacts'. Это дублирует A1 hard gate, потому что Resolver не выдаёт contacts в route_intent.

**Кандидат на унификацию:** расширить CONTACTS_RE в A1 hard gate, overlay убрать. Между PR #1.3 и PR #1.4.

---

## prices.json incomplete (client-specific cesi)

Каталог имеет price_key для 11 услуг, prices.json содержит только 4 (tomography, professional_whitening, teeth_treatment, zirconia_crowns). Это реальные данные клиента cesi, не баг.

Архитектура поддерживает оба пути: price_card (JSON) и price-context (MD через price_ref / default fallback).

**Для следующих клиентов:** проверить покрытие prices.json при онбординге.

---

## All-on-4 / All-on-6 placeholder pricing

В implantation__pricing__implants.md есть раздел #vse-na-4-i-6-tseny с placeholder-ценами 320k / 420k. Это тестовые данные, не реальные.

**Перед прод-деплоем для cesi:** согласовать с клиентом или убрать конкретные цифры.

---

## not_offered_services pattern (no architectural support)

Клиника может явно НЕ предлагать определённые услуги (cesi: брекеты). Сейчас бот это не обрабатывает — может попасть в случайный документ или сгенерировать ответ про несуществующую услугу.

**Кандидат на fix:** новая ветка в A1 hard gates после PR #1.3:

- service_catalog.json: блок not_offered_services с aliases + template
- В _orchestrate_ask_turn: после CONTACTS_RE → проверка not_offered match

**Приоритет:** после PR #1.3–#1.4, до PR B (boosters).
