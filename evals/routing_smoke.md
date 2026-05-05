# P0 routing smoke eval (20–30)

Цель: маленький набор кейсов для проверки routing/regressions до и после фикса `intent=content` (conflict `catalog md_first` vs retrieval).

Формат кейса:
- q
- expected_route
- expected_doc_id или expected_topic
- forbidden_doc_id (опционально)
- note (критерий правильности в 1–2 фразах)

---

## FAQ / specific (retrieval должен побеждать service overview)

### Case 01
- q: какая приживаемость имплантов?
- expected_route: retrieval_chunk
- expected_doc_id: implantation__faq__osseointegration
- forbidden_doc_id: implantation__service__classic
- note: ответ про приживаемость/приживление, без «обзора услуги» и без цены.

### Case 02
- q: риск отторжения импланта
- expected_route: retrieval_chunk
- expected_doc_id: implantation__faq__osseointegration
- forbidden_doc_id: implantation__service__classic
- note: про риски/отторжение/приживление, без цены.

### Case 03
- q: сколько длится приживление импланта?
- expected_route: retrieval_chunk
- expected_doc_id: implantation__faq__duration
- forbidden_doc_id: implantation__service__classic
- note: сроки/этапы приживления, без цены.

### Case 04
- q: больно ли ставить имплант?
- expected_route: retrieval_chunk
- expected_doc_id: implantation__faq__pain
- forbidden_doc_id: implantation__service__classic
- note: про боль/анестезию/ощущения, без цены.

### Case 05
- q: если кости мало, что делать?
- expected_route: retrieval_chunk
- expected_doc_id: implantation__info__bone_graft
- forbidden_doc_id: implantation__service__classic
- note: про костную пластику/наращивание (если в базе так описано), без цены.

### Case 06
- q: какие противопоказания к имплантации?
- expected_route: retrieval_chunk
- expected_doc_id: implantation__info__contraindications
- forbidden_doc_id: implantation__service__classic
- note: перечислить противопоказания/ограничения из базы.

### Case 07
- q: что такое остеоинтеграция?
- expected_route: retrieval_chunk
- expected_doc_id: implantation__faq__osseointegration
- forbidden_doc_id: implantation__service__classic
- note: определить термин простыми словами.

---

## Service overview (catalog md_first может побеждать, если retrieval слабый)

### Case 08
- q: расскажите про классическую имплантацию
- expected_route: catalog_md_first
- expected_doc_id: implantation__service__classic
- note: обзор услуги (коротко), допускается уместная цена.

### Case 09
- q: что такое all-on-4?
- expected_route: catalog_md_first
- expected_doc_id: implantation__service__all_on_4
- note: обзор методики, без ухода в нерелевантный FAQ.

### Case 10
- q: что такое all-on-6?
- expected_route: catalog_md_first
- expected_doc_id: implantation__service__all_on_6
- note: обзор методики.

### Case 11
- q: расскажите про имплантацию за один этап
- expected_route: catalog_md_first
- expected_doc_id: implantation__service__one_stage
- note: обзор услуги/подхода.

---

## Price routing (не должно ломаться арбитром content)

### Case 12
- q: сколько стоит имплантация?
- expected_route: price_lookup
- expected_topic: price
- note: price route, не content.

### Case 13
- q: цена импланта
- expected_route: price_lookup
- expected_topic: price
- note: price route.

### Case 14
- q: почему так дорого?
- expected_route: price_concern
- expected_topic: price
- note: возражение по цене (не content).

---

## Contacts / clinic info (не должно ломаться)

### Case 15
- q: где вы находитесь?
- expected_route: contacts_chunk
- expected_topic: contacts
- note: адрес/контакты.

### Case 16
- q: телефон клиники
- expected_route: contacts_chunk
- expected_topic: contacts
- note: телефон/канал связи.

---

## Doctors (specific doc)

### Case 17
- q: какие у вас врачи имплантологи?
- expected_route: retrieval_chunk
- expected_doc_id: doctors__doctor__overview
- note: список/описание врачей из базы.

---

## Broad / ambiguous content (guided, не падать автоматически в catalog)

### Case 18
- q: импланты
- expected_route: guided
- expected_topic: implantation
- note: короткий обзор + выбор направления (стоимость/больно/сроки/подходит/записаться).

### Case 19
- q: расскажите
- expected_route: guided
- expected_topic: clarify
- note: бот должен уточнить одним шагом, не делать вид что понял.

### Case 20
- q: хочу зубы
- expected_route: guided
- expected_topic: clarify
- note: мягко направить к выбору темы или к заявке.

