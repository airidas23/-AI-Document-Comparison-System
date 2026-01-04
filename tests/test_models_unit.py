from __future__ import annotations

import pytest

from comparison.models import Diff, Style, TextBlock


def test_style_get_fingerprint_has_normalized_keys():
    fp = Style(font="Times New Roman", size=12.4, bold=True, italic=False).get_fingerprint()
    assert fp["weight"] == "bold"
    assert fp["slant"] == "normal"
    assert "font_family_normalized" in fp
    assert "size_bucket" in fp


def test_textblock_bbox_helpers_roundtrip():
    blk = TextBlock(text="x", bbox={"x": 10, "y": 20, "width": 30, "height": 40}, metadata={})

    as_tuple = blk.get_bbox_tuple()
    assert as_tuple == (10, 20, 40, 60)

    norm = blk.normalize_bbox(page_width=100, page_height=200)
    assert 0.0 <= norm["x"] <= 1.0
    assert 0.0 <= norm["y"] <= 1.0

    # Treating existing bbox as normalized dict for denormalization
    blk_norm = TextBlock(text="x", bbox=norm, metadata={})
    x0, y0, x1, y1 = blk_norm.denormalize_bbox(page_width=100, page_height=200)
    assert x0 == pytest.approx(10)
    assert y0 == pytest.approx(20)
    assert x1 == pytest.approx(40)
    assert y1 == pytest.approx(60)


def test_diff_bbox_normalization_paths_and_metadata_storage():
    d = Diff(page_num=1, diff_type="modified", change_type="content", old_text="a", new_text="b", bbox=None)
    assert d.normalize_bbox(100, 100) is None

    # Already-normalized dict should be returned as-is.
    d2 = Diff(
        page_num=1,
        diff_type="modified",
        change_type="content",
        old_text="a",
        new_text="b",
        bbox={"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
    )
    assert d2.normalize_bbox(100, 100) == {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}

    # Absolute dict should be normalized.
    d3 = Diff(
        page_num=1,
        diff_type="modified",
        change_type="content",
        old_text="a",
        new_text="b",
        bbox={"x": 10, "y": 20, "width": 30, "height": 40},
    )
    norm3 = d3.normalize_bbox(100, 200)
    assert norm3 == {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.2}

    # Tuple path.
    d4 = Diff(
        page_num=1,
        diff_type="modified",
        change_type="content",
        old_text="a",
        new_text="b",
        bbox=(10, 20, 40, 60),
    )
    assert d4.normalize_bbox(100, 200) == {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.2}

    # get_normalized_bbox stores page dimensions.
    d5 = Diff(
        page_num=1,
        diff_type="modified",
        change_type="content",
        old_text="a",
        new_text="b",
        bbox={"x": 10, "y": 20, "width": 30, "height": 40},
    )
    n = d5.get_normalized_bbox(100, 200)
    assert n is not None
    assert d5.metadata["page_width"] == 100
    assert d5.metadata["page_height"] == 200

    # get_bbox_tuple works for both tuple + dict
    assert d4.get_bbox_tuple() == (10, 20, 40, 60)
    assert d3.get_bbox_tuple() == (10, 20, 40, 60)
