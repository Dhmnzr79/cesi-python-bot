# P0 routing smoke eval (20–30)

Цель: маленький набор кейсов для проверки routing/regressions (catalog md_first vs retrieval vs price routes).

Формат кейса:
- q
- expected_route
- expected_doc_id или expected_topic
- forbidden_doc_id (опционально)
- note (критерий правильности в 1–2 фразах)

---

## Cross-topic guard / non-specific guard (service question не должен уехать в FAQ/другой topic)

### Case 01
- q: делаете ли вы имплантацию?
- expected_route: catalog_md_first
- expected_doc_id: implantation__info__methods_overview
- forbidden_doc_id: extraction__service__tooth_extraction
- note: должен остаться в overview имплантации; не уезжать в удаление зуба или FAQ по боли.

### Case 02
- q: хочу поставить имплант
- expected_route: catalog_md_first
- expected_doc_id: implantation__info__methods_overview
- forbidden_doc_id: implantation__faq__pain
- note: не должен уходить в страх боли; это запрос “наличие/что делать дальше”.

### Case 03
- q: делаете ли вы коронки?
- expected_route: catalog_md_first
- expected_doc_id: prosthetics__service__zirconia_crowns
- forbidden_doc_id: implantation__service__classic
- note: вопрос про коронки должен вести в ортопедию, не в имплантацию.

### Case 04
- q: лечите ли вы кариес?
- expected_route: catalog_md_first
- expected_doc_id: treatment__service__teeth_treatment
- forbidden_doc_id: implantation__faq__pain
- note: вопрос про лечение зубов не должен уехать в FAQ имплантации из-за “больно/анестезия”.

### Case 05
- q: есть ли у вас виниры?
- expected_route: catalog_md_first
- expected_doc_id: prosthetics__service__veneers
- forbidden_doc_id: prosthetics__service__zirconia_crowns
- note: “есть ли” должен вести в правильную услугу, не в соседнюю по эстетике.

### Case 06
- q: делаете ли вы элайнеры?
- expected_route: catalog_md_first
- expected_doc_id: orthodontics__service__aligners
- forbidden_doc_id: treatment__service__teeth_treatment
- note: вопрос про выравнивание должен вести в элайнеры, не в лечение зубов.

