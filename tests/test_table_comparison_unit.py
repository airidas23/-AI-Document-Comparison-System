import importlib.util

import pytest

from comparison.models import PageData, TextBlock
from comparison.table_comparison import (
    TableStructure,
    _calculate_table_overlap,
    _compare_table_structures,
    _extract_table_structure,
    _filter_contained_tables,
    _match_tables,
    compare_tables,
)


def _fake_fitz_spec(name: str):
    if name == "fitz":
        return object()
    return None


def _page_with_blocks(*, blocks, tables_meta=None, extraction_method="digital"):
    metadata = {"extraction_method": extraction_method}
    if tables_meta is not None:
        metadata["tables"] = tables_meta
    return PageData(page_num=1, width=200, height=200, blocks=blocks, metadata=metadata)


def test_filter_contained_tables_removes_wrapper_and_duplicates():
    wrapper = TableStructure(
        rows=1,
        cols=1,
        cells=[["W"]],
        bbox={"x": 0, "y": 0, "width": 100, "height": 100},
        confidence=0.9,
    )
    inner = TableStructure(
        rows=1,
        cols=1,
        cells=[["I"]],
        bbox={"x": 10, "y": 10, "width": 80, "height": 80},
        confidence=0.9,
    )
    # Near-duplicate of inner (heavily overlapping).
    inner2 = TableStructure(
        rows=1,
        cols=1,
        cells=[["I"]],
        bbox={"x": 11, "y": 11, "width": 79, "height": 79},
        confidence=0.9,
    )

    filtered = _filter_contained_tables([wrapper, inner, inner2])

    # Expect wrapper removed and one of the duplicates removed.
    assert len(filtered) == 1
    assert filtered[0].bbox["x"] in (10, 11)


def test_extract_table_structure_parses_bbox_list_and_numeric_rows(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", _fake_fitz_spec)

    blocks = [
        TextBlock(text="Header1", bbox={"x": 10, "y": 10, "width": 60, "height": 10}),
        TextBlock(text="Header2", bbox={"x": 110, "y": 10, "width": 60, "height": 10}),
        TextBlock(text="Foo 1 2", bbox={"x": 10, "y": 40, "width": 160, "height": 10}),
        TextBlock(text="Bar Baz 3", bbox={"x": 10, "y": 70, "width": 160, "height": 10}),
    ]
    page = _page_with_blocks(blocks=blocks, tables_meta=[])

    region = {"bbox": [0, 0, 200, 200], "confidence": 0.95}
    table = _extract_table_structure(page, region)

    assert table is not None
    assert table.rows == 3
    assert table.cols == 3

    # Header row tokenizes.
    assert table.cells[0] == ["Header1", "Header2", ""]
    # Numeric row keeps label + numeric values.
    assert table.cells[1] == ["Foo", "1", "2"]
    # Multi-word label handled.
    assert table.cells[2] == ["Bar Baz", "3", ""]

    assert table.content_bbox is not None
    assert table.content_insets_norm is not None
    for k in ("left", "top", "right", "bottom"):
        assert -0.05 <= float(table.content_insets_norm[k]) <= 1.0


def test_extract_table_structure_handles_invalid_bbox_and_missing(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", _fake_fitz_spec)

    page = _page_with_blocks(blocks=[], tables_meta=[])

    assert _extract_table_structure(page, [0, 0, 1, 1]) is None
    assert _extract_table_structure(page, {}) is None
    assert _extract_table_structure(page, {"bbox": [0, 0, 1]}) is None
    assert _extract_table_structure(page, {"bbox": {"x": "nope"}}) is None


def test_calculate_table_overlap_no_intersection():
    a = TableStructure(1, 1, [["A"]], {"x": 0, "y": 0, "width": 10, "height": 10}, 0.9)
    b = TableStructure(1, 1, [["B"]], {"x": 20, "y": 20, "width": 10, "height": 10}, 0.9)
    assert _calculate_table_overlap(a, b) == 0.0


def test_match_tables_uses_overlap_threshold(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "table_overlap_threshold", 0.1)

    a = TableStructure(1, 1, [["A"]], {"x": 0, "y": 0, "width": 10, "height": 10}, 0.9)
    b = TableStructure(1, 1, [["B"]], {"x": 1, "y": 1, "width": 10, "height": 10}, 0.9)

    matched, ua, ub = _match_tables([a], [b])
    assert matched
    assert not ua
    assert not ub


def test_compare_table_structures_digital_same_signature_structure_changed_returns_style_visibility():
    # Cells identical, but cols differ -> structure_changed True.
    table_a = TableStructure(
        rows=2,
        cols=2,
        cells=[["H1", "H2"], ["1", "2"]],
        bbox={"x": 0, "y": 0, "width": 100, "height": 100},
        confidence=0.9,
    )
    table_b = TableStructure(
        rows=2,
        cols=3,
        cells=[["H1", "H2"], ["1", "2"]],
        bbox={"x": 0, "y": 0, "width": 100, "height": 100},
        confidence=0.9,
    )

    diffs = _compare_table_structures(
        table_a,
        table_b,
        1,
        1,
        0.9,
        200,
        200,
        200,
        200,
        is_ocr_page=False,
    )

    assert len(diffs) == 1
    assert diffs[0].change_type == "formatting"
    assert diffs[0].metadata.get("table_change") == "style"
    assert diffs[0].metadata.get("subtype") == "structure_visibility"


def test_compare_table_structures_digital_padding_or_border_detected(monkeypatch):
    from config.settings import settings

    # Make thresholds easy to trigger.
    monkeypatch.setattr(settings, "table_style_inset_change_threshold", 0.05)
    monkeypatch.setattr(settings, "table_style_bbox_change_threshold", 0.05)

    table_a = TableStructure(
        rows=1,
        cols=2,
        cells=[["A", "B"]],
        bbox={"x": 0, "y": 0, "width": 100, "height": 100},
        confidence=0.9,
        content_insets_norm={"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0},
    )
    table_b = TableStructure(
        rows=1,
        cols=2,
        cells=[["A", "B"]],
        bbox={"x": 0, "y": 0, "width": 100, "height": 100},
        confidence=0.9,
        content_insets_norm={"left": 0.2, "top": 0.0, "right": 0.0, "bottom": 0.0},
    )

    diffs = _compare_table_structures(
        table_a,
        table_b,
        1,
        1,
        0.9,
        200,
        200,
        200,
        200,
        is_ocr_page=False,
    )

    assert len(diffs) == 1
    assert diffs[0].change_type == "formatting"
    assert diffs[0].metadata.get("subtype") == "padding_or_border"


def test_compare_table_structures_ocr_low_confidence_region_changed(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "ocr_table_structure_confidence_threshold", 0.9)

    table_a = TableStructure(
        rows=1,
        cols=2,
        cells=[["A", "B"]],
        bbox={"x": 0, "y": 0, "width": 100, "height": 100},
        confidence=0.5,
    )
    # Small col diff (±1) should be treated as non-structural when confidence is low.
    table_b = TableStructure(
        rows=1,
        cols=3,
        cells=[["A", "C", "X"]],
        bbox={"x": 0, "y": 0, "width": 100, "height": 100},
        confidence=0.5,
    )

    diffs = _compare_table_structures(
        table_a,
        table_b,
        1,
        1,
        0.9,
        200,
        200,
        200,
        200,
        is_ocr_page=True,
    )

    assert len(diffs) == 1
    assert diffs[0].metadata.get("table_change") == "table_region_changed"
    assert diffs[0].metadata.get("low_confidence") is True


def test_compare_table_structures_ocr_structure_change_allowed_for_stable_col_plus_one(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "ocr_table_structure_confidence_threshold", 0.8)

    table_a = TableStructure(
        rows=1,
        cols=2,
        cells=[["H1", "H2"]],
        bbox={"x": 0, "y": 0, "width": 100, "height": 100},
        confidence=0.95,
    )
    table_b = TableStructure(
        rows=1,
        cols=3,
        cells=[["H1", "H2", "Delta"]],
        bbox={"x": 0, "y": 0, "width": 100, "height": 100},
        confidence=0.95,
    )

    diffs = _compare_table_structures(
        table_a,
        table_b,
        1,
        1,
        0.9,
        200,
        200,
        200,
        200,
        is_ocr_page=True,
    )

    assert len(diffs) == 1
    assert diffs[0].metadata.get("subtype") == "table_columns_changed"


def test_compare_table_structures_ocr_skips_insignificant_cell_noise(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "ocr_table_structure_confidence_threshold", 0.1)

    # Patch OCR significance to always treat changes as insignificant.
    import utils.text_normalization as tn

    def _fake_sig(*args, **kwargs):
        return {"is_significant": False}

    monkeypatch.setattr(tn, "compute_ocr_change_significance", _fake_sig)

    table_a = TableStructure(
        rows=1,
        cols=1,
        cells=[["A"]],
        bbox={"x": 0, "y": 0, "width": 100, "height": 100},
        confidence=0.95,
    )
    table_b = TableStructure(
        rows=1,
        cols=1,
        cells=[["B"]],
        bbox={"x": 0, "y": 0, "width": 100, "height": 100},
        confidence=0.95,
    )

    diffs = _compare_table_structures(
        table_a,
        table_b,
        1,
        1,
        0.9,
        200,
        200,
        200,
        200,
        is_ocr_page=True,
    )

    assert diffs == []


def test_compare_tables_recovers_missing_table_on_one_side(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", _fake_fitz_spec)

    # A has explicit table region metadata.
    blocks_a = [
        TextBlock(text="H1", bbox={"x": 10, "y": 10, "width": 50, "height": 10}),
        TextBlock(text="H2", bbox={"x": 110, "y": 10, "width": 50, "height": 10}),
        TextBlock(text="A 1", bbox={"x": 10, "y": 40, "width": 160, "height": 10}),
    ]
    page_a = _page_with_blocks(
        blocks=blocks_a,
        tables_meta=[{"bbox": [0, 0, 200, 200], "confidence": 0.95}],
        extraction_method="digital",
    )

    # B has no table metadata, but content exists in the same bbox; recovery should kick in.
    blocks_b = [
        TextBlock(text="H1", bbox={"x": 10, "y": 10, "width": 50, "height": 10}),
        TextBlock(text="H2", bbox={"x": 110, "y": 10, "width": 50, "height": 10}),
        TextBlock(text="B 1", bbox={"x": 10, "y": 40, "width": 160, "height": 10}),
    ]
    page_b = _page_with_blocks(blocks=blocks_b, tables_meta=None, extraction_method="digital")

    diffs = compare_tables([page_a], [page_b], alignment_map={1: (1, 0.9)})

    # Should get at least one diff from the changed table content (cell-level compare).
    assert diffs
    assert any(d.metadata.get("type") == "table" for d in diffs)
