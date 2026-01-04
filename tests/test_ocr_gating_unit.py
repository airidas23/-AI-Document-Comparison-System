import math
import sys
import types

import pytest

from comparison.ocr_gating import (
    EmbeddingCache,
    GateDecision,
    GatingConfig,
    GatingResult,
    _compute_bbox_iou,
    _compute_length_ratio,
    _compute_position_drift,
    apply_gating,
    batch_gating,
    clear_embedding_cache,
    collect_gating_stats,
    get_cache_stats,
    stage_a_gate,
    stage_b_semantic,
)


class _FakeNumpy(types.ModuleType):
    def __init__(self):
        super().__init__("numpy")

        class _Linalg:
            @staticmethod
            def norm(vec):
                return math.sqrt(sum(float(v) * float(v) for v in vec))

        self.linalg = _Linalg()

    @staticmethod
    def dot(a, b):
        return sum(float(x) * float(y) for x, y in zip(a, b))


class _FakeSentenceTransformer:
    def __init__(self, embeddings_by_text: dict[str, list[float]]):
        self._embeddings_by_text = embeddings_by_text

    def encode(self, texts, convert_to_numpy=True):
        return [self._embeddings_by_text[t] for t in texts]


class _FakeSentenceTransformersModule(types.ModuleType):
    def __init__(self, embeddings_by_text: dict[str, list[float]]):
        super().__init__("sentence_transformers")

        def SentenceTransformer(_model_name: str):
            return _FakeSentenceTransformer(embeddings_by_text)

        self.SentenceTransformer = SentenceTransformer


def test_compute_length_ratio_edge_cases():
    assert _compute_length_ratio("", "") == 1.0
    assert _compute_length_ratio("abc", "") == 0.0
    assert _compute_length_ratio("a" * 10, "b" * 20) == 0.5


def test_compute_bbox_iou_none_degenerate_and_overlap():
    assert _compute_bbox_iou(None, None) is None
    assert _compute_bbox_iou((0, 0, 0, 10), (0, 0, 10, 10)) is None
    assert _compute_bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    # 10x10 overlap 5x5 => intersection 25
    # area_a=100, area_b=100, union=175 => iou=25/175
    iou = _compute_bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert iou == pytest.approx(25 / 175)


def test_compute_position_drift():
    assert _compute_position_drift(None, None) is None

    drift = _compute_position_drift(
        (0, 0, 10, 10),
        (10, 0, 20, 10),
        page_width=100,
        page_height=100,
    )
    # centers are (5,5) vs (15,5): dx=10/100=0.1, dy=0
    assert drift == pytest.approx(0.1)


def test_stage_a_gate_length_ratio_outside_bounds():
    cfg = GatingConfig(length_ratio_min=0.9, length_ratio_max=1.1)
    res = stage_a_gate("short", "a" * 200, cfg)
    assert res.decision == GateDecision.LIKELY_DIFFERENT
    assert "Length ratio" in res.reason


def test_stage_a_gate_gray_zone_triggers_semantic_on_bbox_drift():
    cfg = GatingConfig(
        identical_threshold=0.999,
        likely_identical_threshold=0.999,
        gray_zone_low=0.0,
        gray_zone_high=1.0,
        length_ratio_min=0.0,
        length_ratio_max=10.0,
        bbox_iou_threshold=0.9,
    )
    res = stage_a_gate(
        "abc def ghi",
        "abc def gni",  # close-ish
        cfg,
        bbox_a=(0, 0, 100, 10),
        bbox_b=(500, 700, 600, 710),
    )
    assert res.decision == GateDecision.SEMANTIC_CHECK


def test_stage_a_gate_triggers_semantic_on_position_drift_when_iou_ok():
    cfg = GatingConfig(
        identical_threshold=0.99,
        likely_identical_threshold=0.92,
        gray_zone_low=0.70,
        gray_zone_high=0.85,
        bbox_iou_threshold=0.5,
        position_drift_threshold=0.05,
    )

    # IoU ~0.667 (>0.5), but drift 40/612 ~0.065 (>0.05)
    res = stage_a_gate(
        "abcdefg",
        "abcxefg",
        cfg,
        bbox_a=(0, 0, 200, 200),
        bbox_b=(40, 0, 240, 200),
    )
    assert res.decision == GateDecision.SEMANTIC_CHECK
    assert res.position_drift is not None
    assert res.position_drift > 0.05


def test_stage_b_semantic_disabled_forces_needs_diff():
    cfg = GatingConfig(enable_stage_b=False)
    stage_a = GatingResult(decision=GateDecision.SEMANTIC_CHECK, confidence=0.75, reason="Gray")
    out = stage_b_semantic("a", "b", stage_a, cfg)
    assert out.decision == GateDecision.NEEDS_DIFF
    assert "Stage B disabled" in out.reason


def test_stage_b_semantic_missing_deps_falls_back_to_needs_diff(monkeypatch):
    # Ensure sentence_transformers import fails.
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", types.SimpleNamespace())

    cfg = GatingConfig(enable_stage_b=True)
    stage_a = GatingResult(decision=GateDecision.SEMANTIC_CHECK, confidence=0.75, reason="Gray")
    out = stage_b_semantic("a", "b", stage_a, cfg)
    assert out.decision == GateDecision.NEEDS_DIFF
    assert out.reason.startswith("Semantic analysis failed:")


def test_stage_b_semantic_cached_paths_and_decisions(monkeypatch):
    from comparison import ocr_gating

    clear_embedding_cache()
    monkeypatch.setitem(sys.modules, "numpy", _FakeNumpy())
    monkeypatch.setitem(sys.modules, "sentence_transformers", _FakeSentenceTransformersModule({}))

    # Populate cache with vectors.
    ocr_gating._embedding_cache.set("A", [1.0, 0.0])
    ocr_gating._embedding_cache.set("B", [1.0, 0.0])
    ocr_gating._embedding_cache.set("C", [1.0, 0.0])
    ocr_gating._embedding_cache.set("D", [0.0, 1.0])

    cfg = GatingConfig(cache_embeddings=True)

    base = GatingResult(decision=GateDecision.SEMANTIC_CHECK, confidence=0.75, reason="Gray")
    out = stage_b_semantic("A", "B", base, cfg)
    assert out.embedding_cached is True
    assert out.decision == GateDecision.LIKELY_IDENTICAL

    base2 = GatingResult(decision=GateDecision.SEMANTIC_CHECK, confidence=0.75, reason="Gray")
    out2 = stage_b_semantic("C", "D", base2, cfg)
    assert out2.embedding_cached is True
    assert out2.decision == GateDecision.LIKELY_DIFFERENT


def test_stage_b_semantic_uncached_computes_and_caches(monkeypatch):
    clear_embedding_cache()

    monkeypatch.setitem(sys.modules, "numpy", _FakeNumpy())

    embeddings_by_text = {
        "hello": [1.0, 0.0],
        "hell0": [0.9, 0.1],
    }
    monkeypatch.setitem(sys.modules, "sentence_transformers", _FakeSentenceTransformersModule(embeddings_by_text))

    cfg = GatingConfig(cache_embeddings=True)
    stage_a = GatingResult(decision=GateDecision.SEMANTIC_CHECK, confidence=0.75, reason="Gray")
    out = stage_b_semantic("hello", "hell0", stage_a, cfg)

    assert out.semantic_score is not None
    assert out.stage_b_time >= 0.0

    stats = get_cache_stats()
    assert stats["size"] > 0


def test_apply_gating_runs_stage_b_when_semantic_check(monkeypatch):
    from comparison import ocr_gating

    def _fake_stage_a(*_a, **_k):
        return GatingResult(decision=GateDecision.SEMANTIC_CHECK, confidence=0.75, reason="gray")

    monkeypatch.setattr(ocr_gating, "stage_a_gate", _fake_stage_a)

    def _fake_stage_b(text_a, text_b, stage_a_result, config):
        stage_a_result.decision = GateDecision.LIKELY_IDENTICAL
        stage_a_result.semantic_score = 0.99
        stage_a_result.reason = "forced"
        return stage_a_result

    monkeypatch.setattr(ocr_gating, "stage_b_semantic", _fake_stage_b)

    res = apply_gating("a", "b", is_ocr=True)
    assert res.decision == GateDecision.LIKELY_IDENTICAL
    assert res.semantic_score == 0.99


def test_batch_gating_splits_skip_and_diff_indices():
    # Covers: both empty -> IDENTICAL, normalized exact match, one empty -> LIKELY_DIFFERENT
    pairs = [("", ""), ("Hello  world ", "Hello world"), ("", "x"), ("abc", "xyz")]
    skip_idx, diff_idx, results = batch_gating(pairs, is_ocr=True)
    assert 0 in skip_idx
    assert 1 in skip_idx
    assert 2 in diff_idx
    assert len(results) == 4


def test_embedding_cache_hit_rate_and_eviction():
    cache = EmbeddingCache(max_size=2)
    cache.set("a", [1])
    assert cache.get("a") == [1]

    cache.set("b", [2])
    cache.set("c", [3])  # triggers eviction
    assert len(cache._cache) <= 2
    assert 0.0 <= cache.hit_rate <= 1.0


def test_collect_gating_stats_counts_and_rates():
    results = [
        GatingResult(decision=GateDecision.IDENTICAL, confidence=1.0, stage_a_time=0.01),
        GatingResult(decision=GateDecision.LIKELY_IDENTICAL, confidence=0.93, stage_a_time=0.01),
        GatingResult(decision=GateDecision.NEEDS_DIFF, confidence=0.5, stage_a_time=0.02),
        GatingResult(decision=GateDecision.LIKELY_DIFFERENT, confidence=0.9, stage_a_time=0.02, semantic_score=0.1),
    ]
    stats = collect_gating_stats(results)
    assert stats.total_pairs == 4
    assert stats.identical == 1
    assert stats.likely_identical == 1
    assert stats.needs_diff == 1
    assert stats.likely_different == 1
    assert stats.semantic_checks == 1
    assert 0.0 <= stats.skip_rate <= 1.0



