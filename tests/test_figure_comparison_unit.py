from __future__ import annotations

import pytest

from comparison.models import PageData, TextBlock


def _blk(text: str, *, x: float, y: float, w: float = 100, h: float = 20) -> TextBlock:
    return TextBlock(text=text, bbox={"x": x, "y": y, "width": w, "height": h}, metadata={})


def _page(page_num: int, *, blocks: list[TextBlock], figures: list[dict]) -> PageData:
    return PageData(
        page_num=page_num,
        width=600.0,
        height=800.0,
        blocks=blocks,
        metadata={"figures": figures},
    )


def test_match_caption_pattern_variants():
    from comparison.figure_comparison import _match_caption_pattern

    assert _match_caption_pattern("Figure 1: Hello") == {"number": 1, "label": "Figure", "full_number": "1"}
    assert _match_caption_pattern("Fig. 2. World")["number"] == 2
    assert _match_caption_pattern("FIG 3-2 Something")["full_number"] == "3.2"
    assert _match_caption_pattern("Not a caption") is None


def test_find_caption_near_figure_picks_closest(monkeypatch):
    from comparison.figure_comparison import _find_caption_near_figure
    from config.settings import settings

    monkeypatch.setattr(settings, "caption_search_margin", 5, raising=False)
    monkeypatch.setattr(settings, "caption_search_distance", 200, raising=False)

    fig_bbox = {"x": 50, "y": 100, "width": 200, "height": 150}

    # Two caption candidates below the figure; the first (smaller y) should be chosen.
    p = _page(
        1,
        blocks=[
            _blk("Figure 1: First", x=60, y=260),
            _blk("Figure 1: Second", x=60, y=300),
        ],
        figures=[{"bbox": [50, 100, 250, 250]}],
    )

    cap = _find_caption_near_figure(p, fig_bbox)
    assert cap and cap["text"].startswith("Figure 1")
    assert cap["text"].endswith("First")


def test_extract_figure_captions_handles_list_bbox_and_missing_caption(monkeypatch):
    from comparison.figure_comparison import extract_figure_captions
    from config.settings import settings

    monkeypatch.setattr(settings, "caption_search_margin", 5, raising=False)
    monkeypatch.setattr(settings, "caption_search_distance", 200, raising=False)

    # First figure has a caption; second does not.
    p = _page(
        1,
        blocks=[_blk("Fig. 2. Hello", x=60, y=260)],
        figures=[
            {"bbox": [50, 100, 250, 250]},
            {"bbox": [300, 100, 500, 250]},
        ],
    )

    caps = extract_figure_captions(p)
    assert len(caps) == 2
    assert caps[0].caption_number == 2
    assert caps[0].caption_label in {"Figure", "Fig."}
    assert caps[1].caption_text == ""


def test_match_figures_by_overlap(monkeypatch):
    from comparison.figure_comparison import FigureCaption, _match_figures
    from config.settings import settings

    monkeypatch.setattr(settings, "figure_overlap_threshold", 0.2, raising=False)

    a1 = FigureCaption(figure_bbox={"x": 10, "y": 10, "width": 100, "height": 100}, caption_text="Figure 1: A", page_num=1)
    a2 = FigureCaption(figure_bbox={"x": 300, "y": 10, "width": 100, "height": 100}, caption_text="Figure 2: B", page_num=1)

    b1 = FigureCaption(figure_bbox={"x": 12, "y": 12, "width": 100, "height": 100}, caption_text="Figure 1: A", page_num=1)

    matched, ua, ub = _match_figures([a1, a2], [b1])
    assert matched and matched[0][0].caption_text.startswith("Figure 1")
    assert len(ua) == 1  # a2 unmatched
    assert len(ub) == 0


def test_compare_figure_captions_added_deleted_and_modified(monkeypatch):
    from comparison import figure_comparison

    # Disable optional image hashing path to keep test deterministic.
    monkeypatch.setattr(figure_comparison, "IMAGEHASH_AVAILABLE", False)

    # Align page 1 -> page 1
    monkeypatch.setattr(figure_comparison, "align_pages", lambda a, b, use_similarity=False: {1: (1, 1.0)})

    from config.settings import settings

    monkeypatch.setattr(settings, "caption_search_margin", 5, raising=False)
    monkeypatch.setattr(settings, "caption_search_distance", 200, raising=False)
    monkeypatch.setattr(settings, "figure_overlap_threshold", 0.2, raising=False)

    # Doc A: one figure with caption
    page_a = _page(
        1,
        blocks=[_blk("Figure 1: Alpha", x=60, y=260)],
        figures=[{"bbox": [50, 100, 250, 250]}],
    )

    # Doc B: same figure location but different number/text + an extra figure
    page_b = _page(
        1,
        blocks=[
            _blk("Figure 2: Beta", x=60, y=260),
            _blk("Figure 3: Extra", x=360, y=260),
        ],
        figures=[
            {"bbox": [52, 102, 252, 252]},
            {"bbox": [300, 100, 500, 250]},
        ],
    )

    diffs = figure_comparison.compare_figure_captions([page_a], [page_b])

    # Should include numbering change, caption text change, and an added figure.
    changes = {(d.metadata.get("figure_change"), d.diff_type, d.change_type) for d in diffs}
    assert ("numbering", "modified", "visual") in changes
    assert ("caption_text", "modified", "content") in changes
    assert ("figure_added", "added", "content") in changes

    # Now flip: missing figure in B -> deletion
    diffs2 = figure_comparison.compare_figure_captions([page_b], [page_a])
    changes2 = {(d.metadata.get("figure_change"), d.diff_type) for d in diffs2}
    assert ("figure_deleted", "deleted") in changes2
