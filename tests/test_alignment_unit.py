from __future__ import annotations

from comparison.models import PageData, TextBlock


def _blk(text: str, *, x: float = 0, y: float = 0, w: float = 100, h: float = 20) -> TextBlock:
    return TextBlock(text=text, bbox={"x": x, "y": y, "width": w, "height": h}, metadata={})


def _page(page_num: int, blocks: list[TextBlock], *, md: dict | None = None) -> PageData:
    return PageData(page_num=page_num, width=600.0, height=800.0, blocks=blocks, metadata=dict(md or {}))


def test_candidate_stats_reset_and_skip_ratio():
    from comparison import alignment

    alignment.reset_candidate_stats()

    # One skipped by length ratio
    ok, reason = alignment.prefilter_candidate_pair("a" * 2, "b" * 100, use_ngram=False, use_bbox=False)
    assert ok is False
    assert reason == "length_ratio"

    stats = alignment.get_candidate_stats()
    assert stats.total_pairs_considered == 1
    assert stats.pairs_skipped_length_ratio == 1
    assert stats.skip_ratio == 1.0

    alignment.reset_candidate_stats()
    stats = alignment.get_candidate_stats()
    assert stats.total_pairs_considered == 0
    assert stats.skip_ratio == 0.0
    d = stats.to_dict()
    assert d["skip_ratio"] == 0.0


def test_prefilter_ngram_hash_and_bbox_distance_skip(monkeypatch):
    from comparison import alignment

    import builtins

    alignment.reset_candidate_stats()

    # Make hash deterministic to avoid any cross-run randomness.
    monkeypatch.setattr(builtins, "hash", lambda s: sum(ord(c) for c in s))

    ok, reason = alignment.prefilter_candidate_pair(
        "abcdefg",  # 3-grams: abc,bcd,cde,def,efg
        "uvwxyz1",  # disjoint
        use_ngram=True,
        use_bbox=False,
        ngram_threshold=0.9,
    )
    assert ok is False
    assert reason == "ngram_hash"

    # BBox distance skip
    ok, reason = alignment.prefilter_candidate_pair(
        "same length",
        "same length",
        bbox_a={"x": 0, "y": 0, "width": 10, "height": 10},
        bbox_b={"x": 10_000, "y": 10_000, "width": 10, "height": 10},
        use_ngram=False,
        use_bbox=True,
        bbox_distance_max=100.0,
    )
    assert ok is False
    assert reason == "bbox_distance"


def test_align_pages_similarity_finds_better_match():
    from comparison.alignment import align_pages

    doc_a = [
        _page(1, [_blk("intro apples")]),
        _page(2, [_blk("conclusion bananas")]),
    ]

    # 3 pages in B: positional match for page 2 would be index 1, but similarity should prefer index 2.
    doc_b = [
        _page(10, [_blk("unrelated")]),
        _page(11, [_blk("intro apples")]),
        _page(12, [_blk("conclusion bananas")]),
    ]

    mapping = align_pages(doc_a, doc_b, use_similarity=True)
    assert mapping[1][0] == 11
    assert mapping[2][0] == 12


def test_find_best_page_match_empty_text_falls_back():
    from comparison.alignment import _find_best_page_match

    page_a = _page(1, [_blk(" ")])
    doc_b = [_page(1, [_blk("x")]), _page(2, [_blk("y")])]

    idx, conf = _find_best_page_match(page_a, doc_b, start_idx=1, end_idx=2)
    assert idx == 1
    assert conf == 0.5


def test_align_sections_hierarchical_failure_falls_back(monkeypatch):
    from comparison import alignment

    # Force hierarchical alignment branch to throw, then verify positional fallback works.
    monkeypatch.setattr(alignment, "hierarchical_align", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))

    from config.settings import settings

    monkeypatch.setattr(settings, "block_alignment_distance_threshold", 1000.0, raising=False)
    monkeypatch.setattr(settings, "ocr_translation_estimation_min_similarity", 0.85, raising=False)
    monkeypatch.setattr(settings, "ocr_min_text_similarity_for_match", 0.0, raising=False)

    page_a = _page(1, [_blk("Hello", x=10, y=10)], md={"has_markdown_structure": True, "extraction_method": "ocr_tesseract"})
    page_b = _page(1, [_blk("Hello", x=20, y=20)], md={"has_markdown_structure": True, "extraction_method": "ocr_tesseract"})

    m = alignment.align_sections(page_a, page_b, use_hierarchical=True)
    assert m == {0: 0}
    assert "page_alignment_translation" in page_a.metadata


def test_estimate_page_translation_and_detect_layout_shift():
    from comparison.alignment import _estimate_page_translation, detect_layout_shift

    # Three strong anchor pairs -> translation estimated.
    blocks_a = [
        _blk("anchor one", x=10, y=10),
        _blk("anchor two", x=10, y=50),
        _blk("anchor three", x=10, y=90),
    ]
    blocks_b = [
        _blk("anchor one", x=15, y=20),
        _blk("anchor two", x=15, y=60),
        _blk("anchor three", x=15, y=100),
    ]

    page_a = _page(1, blocks_a)
    page_b = _page(1, blocks_b)

    dx, dy, conf = _estimate_page_translation(page_a, page_b, min_similarity=0.99)
    assert dx != 0.0
    assert dy != 0.0
    assert conf > 0.0

    # With translation compensation, a small move should not count as a layout shift.
    b0 = _blk("x", x=100, y=100, w=50, h=10)
    b1 = _blk("x", x=105, y=110, w=50, h=10)
    assert (
        detect_layout_shift(
            b0,
            b1,
            page_width=1000,
            page_height=1000,
            tolerance_ratio=0.01,
            translation={"dx": 5.0, "dy": 10.0},
        )
        is None
    )

    # Without translation, the same change exceeds tolerance.
    shift = detect_layout_shift(b0, b1, page_width=1000, page_height=1000, tolerance_ratio=0.001)
    assert shift and shift["shift_detected"] is True
