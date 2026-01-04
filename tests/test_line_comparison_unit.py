"""Unit tests for line-level comparison helpers.

Focused on deterministic, pure-function branches (no OCR engines, no PDF rendering).
"""

from __future__ import annotations

import sys

import pytest

from comparison.models import Line, PageData, Token


def _mk_line(
    line_id: str,
    text: str,
    *,
    x: float = 10,
    y: float = 10,
    w: float = 100,
    h: float = 10,
    reading_order: int = 0,
    tokens: list[Token] | None = None,
) -> Line:
    return Line(
        line_id=line_id,
        bbox={"x": x, "y": y, "width": w, "height": h},
        text=text,
        confidence=1.0,
        reading_order=reading_order,
        tokens=list(tokens or []),
        metadata={},
    )


def test_is_in_header_footer_band_uses_page_height_and_settings(monkeypatch):
    from comparison import line_comparison

    page = PageData(page_num=1, width=600, height=800)

    monkeypatch.setattr(line_comparison.settings, "header_region_height_ratio", 0.10, raising=False)
    monkeypatch.setattr(line_comparison.settings, "footer_region_height_ratio", 0.10, raising=False)

    header_line = _mk_line("h", "hdr", y=5)
    middle_line = _mk_line("m", "mid", y=200)
    footer_line = _mk_line("f", "ftr", y=790, h=15)

    assert line_comparison._is_in_header_footer_band(header_line, page) is True
    assert line_comparison._is_in_header_footer_band(middle_line, page) is False
    assert line_comparison._is_in_header_footer_band(footer_line, page) is True


def test_punctuation_only_change_detects_only_punct_diffs():
    from comparison.line_comparison import _punctuation_only_change

    assert _punctuation_only_change("Hello.", "Hello") is True
    assert _punctuation_only_change("Hello, world!", "Hello world") is True
    assert _punctuation_only_change("Hello", "Hello") is False
    assert _punctuation_only_change("Hello", "Hallo") is False


def test_line_changed_marks_punctuation_only_when_normalized_equal(monkeypatch):
    from comparison import line_comparison

    # Force normalized equality so the punctuation-only branch is exercised.
    monkeypatch.setattr(line_comparison, "normalize_text", lambda *_a, **_k: "same")

    changed, info = line_comparison.line_changed("Hello.", "Hello", is_ocr_page=False)
    assert changed is True
    assert info == {"punctuation_only": True}


def test_line_changed_ocr_filters_insignificant_noise(monkeypatch):
    from comparison import line_comparison

    # Ensure normalized texts differ so the OCR-significance path runs.
    monkeypatch.setattr(line_comparison, "normalize_text", lambda text, ocr=False: str(text))

    monkeypatch.setattr(
        line_comparison,
        "compute_ocr_change_significance",
        lambda *_a, **_k: {"is_significant": False, "score": 0.01},
    )

    changed, info = line_comparison.line_changed("a", "b", is_ocr_page=True)
    assert changed is False
    assert info and info["is_significant"] is False


def test_line_changed_ocr_reports_significant_change(monkeypatch):
    from comparison import line_comparison

    # Ensure normalized texts differ so the OCR-significance path runs.
    monkeypatch.setattr(line_comparison, "normalize_text", lambda text, ocr=False: str(text))

    monkeypatch.setattr(
        line_comparison,
        "compute_ocr_change_significance",
        lambda *_a, **_k: {"is_significant": True, "score": 0.9},
    )

    changed, info = line_comparison.line_changed("a", "b", is_ocr_page=True)
    assert changed is True
    assert info and info["is_significant"] is True


def test_parse_region_bbox_supports_dict_and_list():
    from comparison.line_comparison import _parse_region_bbox

    assert _parse_region_bbox({"bbox": {"x": 1, "y": 2, "width": 3, "height": 4}}) == {
        "x": 1.0,
        "y": 2.0,
        "width": 3.0,
        "height": 4.0,
    }

    assert _parse_region_bbox({"bbox": [10, 20, 30, 40]}) == {
        "x": 10.0,
        "y": 20.0,
        "width": 20.0,
        "height": 20.0,
    }

    assert _parse_region_bbox({"bbox": None}) is None
    assert _parse_region_bbox("nope") is None


def test_line_in_table_region_uses_center_point():
    from comparison.line_comparison import _line_in_table_region

    page = PageData(page_num=1, width=600, height=800)
    page.metadata = {
        "tables": [
            {"bbox": {"x": 0, "y": 0, "width": 100, "height": 100}},
        ]
    }

    inside = _mk_line("l1", "x", x=10, y=10, w=20, h=20)
    outside = _mk_line("l2", "y", x=200, y=200, w=20, h=20)

    assert _line_in_table_region(inside, page) is True
    assert _line_in_table_region(outside, page) is False


def test_line_layout_shift_translation_compensation():
    from comparison.line_comparison import _line_layout_shift

    a = _mk_line("a", "t", x=10, y=10, w=100, h=10)
    b = _mk_line("b", "t", x=30, y=10, w=100, h=10)  # dx=20

    # Without translation, we should detect a shift.
    assert _line_layout_shift(a, b, page_width=600, page_height=800, tolerance_ratio=0.01) is not None

    # With translation that cancels dx, we should not detect a shift.
    assert (
        _line_layout_shift(
            a,
            b,
            page_width=600,
            page_height=800,
            tolerance_ratio=0.01,
            translation={"dx": 20, "dy": 0},
        )
        is None
    )


def test_compute_word_level_highlight_basic_replace():
    from comparison.line_comparison import _compute_word_level_highlight

    page_a = PageData(page_num=1, width=600, height=800)
    page_b = PageData(page_num=1, width=600, height=800)

    t1a = Token(token_id="t1", bbox={"x": 10, "y": 10, "width": 30, "height": 10}, text="Hello")
    t2a = Token(token_id="t2", bbox={"x": 45, "y": 10, "width": 30, "height": 10}, text="world")

    t1b = Token(token_id="t1", bbox={"x": 10, "y": 10, "width": 30, "height": 10}, text="Hello")
    # Make a real token change (not just casing), otherwise matching normalizes it away.
    t2b = Token(token_id="t2", bbox={"x": 45, "y": 10, "width": 30, "height": 10}, text="earth")

    line_a = _mk_line("a", "Hello world", tokens=[t1a, t2a])
    line_b = _mk_line("b", "Hello earth", tokens=[t1b, t2b])

    out = _compute_word_level_highlight(line_a, line_b, page_a, page_b)
    assert out["highlight_mode"] in ("word", "line_fallback")
    assert out.get("word_ops")


def test_compute_word_level_highlight_can_drop_punctuation_tokens():
    from comparison.line_comparison import _compute_word_level_highlight

    page_a = PageData(page_num=1, width=600, height=800)
    page_b = PageData(page_num=1, width=600, height=800)

    tok_a = [
        Token(token_id="t1", bbox={"x": 10, "y": 10, "width": 30, "height": 10}, text="Hello"),
        Token(token_id="t2", bbox={"x": 45, "y": 10, "width": 5, "height": 10}, text=","),
        Token(token_id="t3", bbox={"x": 55, "y": 10, "width": 30, "height": 10}, text="world"),
    ]
    tok_b = [
        Token(token_id="t1", bbox={"x": 10, "y": 10, "width": 30, "height": 10}, text="Hello"),
        Token(token_id="t2", bbox={"x": 45, "y": 10, "width": 5, "height": 10}, text=";"),
        Token(token_id="t3", bbox={"x": 55, "y": 10, "width": 30, "height": 10}, text="world"),
    ]

    line_a = _mk_line("a", "Hello , world", tokens=tok_a)
    line_b = _mk_line("b", "Hello ; world", tokens=tok_b)

    out = _compute_word_level_highlight(
        line_a,
        line_b,
        page_a,
        page_b,
        allow_punctuation_tokens=False,
    )

    # If we drop punctuation tokens, we may end up with no bboxes.
    assert out["highlight_mode"] in ("word", "line_fallback")


def test_merge_line_group_combines_text_and_bbox():
    from comparison.line_comparison import _merge_line_group

    l1 = _mk_line("l1", "a", x=10, y=10, w=10, h=10)
    l2 = _mk_line("l2", "b", x=30, y=10, w=20, h=10)

    merged = _merge_line_group([l1, l2])
    assert merged.metadata.get("is_merged_paragraph") is True
    assert merged.text == "a b"
    assert merged.bbox["x"] == 10
    assert merged.bbox["width"] >= 40 - 10


def test_merge_lines_to_paragraphs_gap_threshold(monkeypatch):
    from comparison import line_comparison

    monkeypatch.setattr(line_comparison.settings, "ocr_paragraph_gap_threshold", 0.1, raising=False)

    l1 = _mk_line("l1", "first", y=10, h=10, reading_order=0)
    # Gap: prev_bottom=20, curr_top=40 -> gap=20; threshold*prev_height=1 -> split.
    l2 = _mk_line("l2", "second", y=40, h=10, reading_order=1)

    merged = line_comparison._merge_lines_to_paragraphs([l1, l2], page=None)
    assert len(merged) == 2

