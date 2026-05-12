"""PR #1.10 alias pipeline regression (no live embed API required for core cases)."""

from __future__ import annotations

import pytest

from core.routing_loader import load_thresholds


def test_select_chunk_no_candidates_no_unbound(monkeypatch: pytest.MonkeyPatch) -> None:
    import query_selector as qs

    monkeypatch.setattr(qs, "retrieve", lambda *a, **k: [])
    out = qs.select_chunk_for_question("anything", client_id="c", sid=None, scope_topic=None)
    assert out["mode"] == "no_candidates"
    assert isinstance(out.get("debug_meta"), dict)


def test_alias_thresholds_present() -> None:
    a = load_thresholds(force_reload=True).alias
    assert a.embedding_high_min == pytest.approx(0.78)
    assert a.strong_effective_min == pytest.approx(0.82)
    assert a.scope_guard_min == pytest.approx(0.85)


def test_deterministic_exact_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    import retriever as r

    r._ALIAS_EMB_MATRIX = None
    r._ALIAS_ROW_CORPUS_IDX = None
    r._ALIAS_ROW_CLIENT = None
    r._ALIAS_ARTIFACTS_ERROR = None

    fake = [
        {
            "file": "demo__service__test.md",
            "client_id": "test_client",
            "aliases": ["all on 4", "олл он 4"],
            "h2": "",
            "h3": "",
            "h2_id": "",
            "h3_id": "korotko",
            "text": "body",
        }
    ]
    monkeypatch.setattr(r, "load_corpus_if_needed", lambda: fake)
    monkeypatch.setattr(r, "_ALIAS_INDEX", r._build_alias_index(fake))

    def _noop_load_alias() -> None:
        import numpy as np

        r._ALIAS_EMB_MATRIX = np.zeros((0, 8), dtype=np.float32)
        r._ALIAS_ROW_CORPUS_IDX = np.array([], dtype=np.int32)
        r._ALIAS_ROW_CLIENT = []
        r._ALIAS_ARTIFACTS_ERROR = ""

    monkeypatch.setattr(r, "_load_alias_embed_artifacts", _noop_load_alias)
    monkeypatch.setattr(r, "_legacy_shadow_enabled", lambda: False)

    out = r.run_alias_pipeline("all on 4", client_id="test_client")
    assert out["diag"]["alias_exact_hit"] is True
    assert out["diag"]["alias_decision"] == "exact"
    assert out["effective_score"] == pytest.approx(1.0)


def test_scope_guard_uses_alias_yaml_not_retrieval_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Conflict guard must read THRESHOLDS.alias.scope_guard_min (PR #1.10)."""
    import query_selector as qs
    import retriever as r

    r._ALIAS_EMB_MATRIX = None
    r._ALIAS_ROW_CORPUS_IDX = None
    r._ALIAS_ROW_CLIENT = None
    r._ALIAS_ARTIFACTS_ERROR = None

    fake = [
        {
            "file": "x.md",
            "client_id": "c",
            "aliases": ["unique_scope_guard_alias_token"],
            "h2": "",
            "h3": "",
            "h2_id": "",
            "h3_id": "k",
            "text": "t",
        }
    ]
    monkeypatch.setattr(r, "load_corpus_if_needed", lambda: fake)
    monkeypatch.setattr(r, "_ALIAS_INDEX", r._build_alias_index(fake))

    def _noop_load_alias() -> None:
        import numpy as np

        r._ALIAS_EMB_MATRIX = np.zeros((0, 8), dtype=np.float32)
        r._ALIAS_ROW_CORPUS_IDX = np.array([], dtype=np.int32)
        r._ALIAS_ROW_CLIENT = []
        r._ALIAS_ARTIFACTS_ERROR = ""

    monkeypatch.setattr(r, "_load_alias_embed_artifacts", _noop_load_alias)
    monkeypatch.setattr(r, "_legacy_shadow_enabled", lambda: False)

    st, reason = qs.compute_retrieval_scope_with_conflict_guard(
        scope_topic_candidate="implantation",
        q="unique_scope_guard_alias_token",
        client_id="c",
    )
    assert st is None
    assert reason == "alias_hit"


def test_legacy_shadow_module_has_expected_api() -> None:
    import alias_scorer_legacy_shadow as leg

    assert hasattr(leg, "corpus_alias_leader_legacy")
    assert hasattr(leg, "legacy_chunk_key")


def test_build_index_alias_norm_matches_retriever_norm() -> None:
    import build_index as bi
    import retriever as r

    s = 'Олл-он 4 {#x#}  !!'
    assert bi._norm_alias_key(s) == r._norm_text(s)
