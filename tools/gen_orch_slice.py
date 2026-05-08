from __future__ import annotations

"""
Codegen PR #1.2.5: AST-splice `_orchestrate_ask_turn` + thin `/ask` into app.py.
Run from repo root: python tools/gen_orch_slice.py
"""

import ast
import textwrap
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"


def _merge_call(call: ast.Call) -> dict:
    """Positional indexes 0,1,... plus keyword names."""
    out: dict = {}
    for i, a in enumerate(call.args):
        out[i] = a
    for k in call.keywords:
        out[k.arg] = k.value
    return out


def _rewrite_return(decision_dump: str, node: ast.Return) -> ast.stmt:
    v = node.value

    if isinstance(v, ast.Tuple) and len(v.elts) == 2:
        first, sec = v.elts
        if isinstance(first, ast.Call) and isinstance(first.func, ast.Name):
            if first.func.id == "jsonify":
                return ast.parse(
                    "return AskOrchestrationResult(\n"
                    "    kind='unknown_client',\n"
                    "    client_error={'error': 'unknown_client'},\n"
                    "    http_status=403,\n)\n",
                ).body[0]
            if first.func.id == "_service_reply" and isinstance(sec, ast.Constant) and sec.value == 429:
                return ast.parse(
                    "return AskOrchestrationResult(\n"
                    "    kind='service_reply',\n"
                    "    q=q,\n"
                    "    sid=sid,\n"
                    "    client_id=client_id,\n"
                    "    service_payload=_rate_limited_response_payload(),\n"
                    "    service_route='rate_limited',\n"
                    "    http_status=429,\n"
                    ")\n",
                ).body[0]
            raise ValueError(f"unsupported 2-tuple return {first.func.id!r}")

    if not isinstance(v, ast.Call) or not isinstance(v.func, ast.Name):
        raise ValueError(ast.unparse(v))

    MKW = _merge_call(v)
    fn = v.func.id

    if fn == "jsonify":
        return ast.parse(
            "return AskOrchestrationResult(\n"
            "    kind='unknown_client',\n"
            "    client_error={'error': 'unknown_client'},\n"
            "    http_status=403,\n)\n",
        ).body[0]

    if fn == "safe_jsonify":
        return ast.parse(
            "return AskOrchestrationResult(\n"
            "    kind='reset_session',\n"
            "    q=q,\n"
            "    sid=sid,\n"
            "    client_id=client_id,\n)\n",
        ).body[0]

    if fn == "_service_reply":
        payload = MKW.get(0) if 0 in MKW else MKW.get("payload")
        if payload is None:
            raise ValueError(ast.unparse(v))
        doc_id_kw = MKW.get("doc_id")
        track_kw = MKW.get("track_user")
        route_kw = MKW.get("route")

        bits = [
            "return AskOrchestrationResult(",
            "    kind='service_reply',",
            "    q=q,",
            "    sid=sid,",
            "    client_id=client_id,",
            f"    service_payload={ast.unparse(payload)},",
        ]
        if doc_id_kw is None:
            bits.append("    service_doc_id=None,")
        else:
            bits.append(f"    service_doc_id={ast.unparse(doc_id_kw)},")
        if track_kw is None:
            bits.append("    service_track_user=True,")
        else:
            bits.append(f"    service_track_user={ast.unparse(track_kw)},")
        if route_kw is None:
            bits.append("    service_route=None,")
        else:
            bits.append(f"    service_route={ast.unparse(route_kw)},")
        bits.append(f"    decision_frame={decision_dump},")
        bits.append(")\n")
        return ast.parse("\n".join(bits)).body[0]

    if fn == "respond_from_chunk":
        dkw = MKW.copy()
        for drop in ("finalize_ask", "safe_jsonify", "logger"):
            dkw.pop(drop, None)
        chunk_e = dkw.pop("chunk", None)
        if chunk_e is None and 0 in dkw:
            chunk_e = dkw.pop(0)
        if chunk_e is None:
            raise ValueError(ast.unparse(v))
        dkw.pop("q", None)
        dkw.pop("sid", None)
        dkw.pop("client_id", None)

        llm_q = dkw.pop("llm_question", None)
        log_ev = dkw.pop("log_event", None)
        route_kw = dkw.pop("route", None)
        if dkw:
            raise ValueError(f"unexpected respond_from_chunk kwargs {list(dkw.keys())}")

        llm_s = ast.unparse(llm_q) if llm_q is not None else "None"
        log_s = ast.unparse(log_ev) if log_ev is not None else '"Answer generated"'
        route_s = ast.unparse(route_kw) if route_kw is not None else '"retrieval_chunk"'

        tpl = "".join(
            [
                "return AskOrchestrationResult(\n",
                "    kind='chunk',\n",
                "    q=q,\n",
                "    sid=sid,\n",
                "    client_id=client_id,\n",
                f"    chosen_chunk={ast.unparse(chunk_e)},\n",
                f"    llm_question={llm_s},\n",
                f"    log_event={log_s},\n",
                f"    chunk_route={route_s},\n",
                f"    decision_frame={decision_dump},\n",
                ")\n",
            ]
        )
        return ast.parse(tpl).body[0]

    raise ValueError(f"unsupported return call {fn!r}\n {ast.unparse(v)}")


class _Rewrite(ast.NodeTransformer):
    def __init__(self, decision_dump: str) -> None:
        self.dd = decision_dump

    def visit_Return(self, node: ast.Return) -> ast.stmt | ast.Return:
        return ast.fix_missing_locations(_rewrite_return(self.dd, node))


def _strip_redundant_decision_none(body: list[ast.stmt]) -> list[ast.stmt]:
    """Remove resolver-prefixed `decision = None` (still have top-level decision=None)."""

    def is_dec_none(st: ast.stmt) -> bool:
        if not isinstance(st, ast.Assign):
            return False
        if len(st.targets) != 1:
            return False
        tg = st.targets[0]
        return isinstance(tg, ast.Name) and tg.id == "decision" and isinstance(st.value, ast.Constant) and st.value.value is None

    def is_ann_safety_net(st: ast.stmt) -> bool:
        return isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id == "safety_net_used"

    out: list[ast.stmt] = []
    i = 0
    while i < len(body):
        cur = body[i]
        if is_dec_none(cur) and i + 1 < len(body) and is_ann_safety_net(body[i + 1]):
            out.append(body[i + 1])
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def main() -> None:
    raw = APP.read_text(encoding="utf-8").replace("\r\n", "\n")

    blob_lines = raw.split("\n")[877:1486]
    chunked = "\n".join(blob_lines)

    shaved_lines: list[str] = []
    for ln in chunked.split("\n"):
        if not ln.strip():
            shaved_lines.append("")
        elif ln.startswith("        "):
            shaved_lines.append(ln[8:])
        else:
            raise AssertionError(repr(ln[:120]))

    inner = "\n".join(shaved_lines)
    src = "decision = None\n" + inner

    mod = ast.parse("def fake(data: dict):\n" + textwrap.indent(src + "\n", "    "))
    fn = mod.body[0]
    assert isinstance(fn, ast.FunctionDef)
    fn.name = "_orchestrate_ask_turn"

    dd = "_orch_decision_dump(decision)"
    fn.body = _strip_redundant_decision_none(fn.body)
    fn.body = [_Rewrite(dd).visit(st) for st in fn.body]
    orch_src = ast.unparse(fn) + "\n"

    dispatch_json = '''

def _orch_decision_dump(decision):
    """DecisionFrame после Resolver либо None (RESOLVER_OFF / ранний выход)."""
    return decision.model_dump() if decision is not None else None


def _dispatch_orchestration_json(orch_r: AskOrchestrationResult):
    """JSON-ответ для /ask (как до рефакторинга)."""
    if orch_r.kind == "unknown_client":
        return jsonify(orch_r.client_error or {"error": "unknown_client"}), orch_r.http_status
    if orch_r.kind == "reset_session":
        return safe_jsonify(reset_session_response(orch_r.sid))
    if orch_r.kind == "service_reply":
        resp = _service_reply(
            orch_r.service_payload,
            orch_r.sid,
            orch_r.q,
            doc_id=orch_r.service_doc_id,
            track_user=orch_r.service_track_user,
            route=orch_r.service_route,
        )
        if orch_r.http_status != 200:
            return resp, orch_r.http_status
        return resp
    if orch_r.kind == "chunk":
        return respond_from_chunk(
            chunk=orch_r.chosen_chunk,
            q=orch_r.q,
            sid=orch_r.sid,
            client_id=orch_r.client_id,
            finalize_ask=finalize_ask,
            safe_jsonify=safe_jsonify,
            logger=logger,
            llm_question=orch_r.llm_question,
            log_event=orch_r.log_event,
            route=orch_r.chunk_route,
        )
    raise RuntimeError(f"bad orchestration kind: {orch_r.kind}")
'''

    ask_endpoint = '''

@app.post("/ask")
def ask():
    q = ""
    request.ctx["turn_t0_monotonic"] = time.monotonic()
    try:
        data = request.get_json(force=True) or {}
        orch_r = _orchestrate_ask_turn(data)
        q = orch_r.q or ""
        return _dispatch_orchestration_json(orch_r)
    except Exception as e:
        logger.exception("ask_failed", extra={"q": q, "err": str(e)})
        if request.ctx.get("sid") and (q or "").strip():
            emit_bot_event(
                logger,
                "turn_complete",
                status="error",
                details={
                    "turn_number": None,
                    "user_text_redacted": redact_text((q or ""), max_len=8000),
                    "user_preview_redacted": redact_text((q or ""), max_len=200),
                    "bot_text_redacted": "",
                    "intent": None,
                    "doc_id": None,
                    "route": "error",
                    "low_score": False,
                    "lead_flow": False,
                    "handoff_filter": False,
                    "answer_chars": 0,
                    "latency_ms": None,
                    "fallback_reason": "ask_failed",
                },
            )
        emit_bot_event(
            logger,
            "ask_failed",
            status="error",
            details={"error": str(e)[:500], "question_preview": (q or "")[:200]},
        )
        return safe_jsonify(internal_error_response()), 200
'''

    new_block = "\n\n".join([orch_src.rstrip(), dispatch_json.strip(), ask_endpoint.strip()]) + "\n"

    start = raw.index('@app.post("/ask")')
    end = raw.index("\n_SSE_HEADERS = {")
    out = raw[:start] + new_block + raw[end:]
    APP.write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
