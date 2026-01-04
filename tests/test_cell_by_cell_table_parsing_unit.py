from __future__ import annotations

from comparison.models import TextBlock


def _blk(text: str, *, x: float, y: float, w: float = 50, h: float = 10) -> TextBlock:
    return TextBlock(text=text, bbox={"x": x, "y": y, "width": w, "height": h}, metadata={})


def test_parse_table_structure_markdown_table():
    from comparison.cell_by_cell_table import parse_table_structure

    blocks = [
        _blk("| A | B |", x=10, y=10),
        _blk("| 1 | 2 |", x=10, y=30),
    ]

    table = parse_table_structure(blocks, page_width=200, page_height=200)
    assert table is not None
    assert table.rows == 2
    assert table.cols == 2
    assert table.headers == ["A", "B"]
    assert table.cells[1][0].text == "1"
    assert table.cells[1][1].text == "2"
    assert table.bbox is not None
    assert table.bbox["width"] > 0


def test_parse_table_structure_html_table():
    from comparison.cell_by_cell_table import parse_table_structure

    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    blocks = [
        _blk(html, x=10, y=10),
        _blk("A", x=10, y=10),
        _blk("B", x=70, y=10),
        _blk("1", x=10, y=30),
        _blk("2", x=70, y=30),
    ]

    table = parse_table_structure(blocks, page_width=200, page_height=200)
    assert table is not None
    assert table.rows >= 2
    assert table.cols == 2
    assert table.headers == ["A", "B"]


def test_parse_table_structure_heuristic_fallback():
    from comparison.cell_by_cell_table import parse_table_structure

    blocks = [
        _blk("H1", x=10, y=10),
        _blk("H2", x=70, y=10),
        _blk("R1C1", x=10, y=50),
        _blk("R1C2", x=70, y=50),
    ]

    table = parse_table_structure(blocks, page_width=200, page_height=200)
    assert table is not None
    assert table.rows >= 2
    assert table.cols == 2
    assert table.headers == ["H1", "H2"]


def test_compute_overlap_and_cluster_by_proximity():
    from comparison.cell_by_cell_table import _cluster_by_proximity, _compute_overlap

    a = {"x": 0, "y": 0, "width": 10, "height": 10}
    b = {"x": 5, "y": 5, "width": 10, "height": 10}
    c = {"x": 20, "y": 20, "width": 5, "height": 5}

    assert _compute_overlap(a, b) > 0.0
    assert _compute_overlap(a, c) == 0.0

    clusters = _cluster_by_proximity([0, 0.4, 5, 5.2, 20], threshold=0.5)
    assert clusters == [[0, 0.4], [5, 5.2], [20]]


def test_detect_column_mapping_matches_headers_and_falls_back():
    from comparison.cell_by_cell_table import ParsedTable, TableCell as ParsedCell, _detect_column_mapping

    # Header-based reordering: A,B,C -> B,C,A
    a = ParsedTable(
        rows=2,
        cols=3,
        headers=["A", "B", "C"],
        cells=[
            [ParsedCell(0, 0, "A"), ParsedCell(0, 1, "B"), ParsedCell(0, 2, "C")],
            [ParsedCell(1, 0, "1"), ParsedCell(1, 1, "2"), ParsedCell(1, 2, "3")],
        ],
    )
    b = ParsedTable(
        rows=2,
        cols=3,
        headers=["B", "C", "A"],
        cells=[
            [ParsedCell(0, 0, "B"), ParsedCell(0, 1, "C"), ParsedCell(0, 2, "A")],
            [ParsedCell(1, 0, "2"), ParsedCell(1, 1, "3"), ParsedCell(1, 2, "1")],
        ],
    )
    mapping = _detect_column_mapping(a, b)
    assert mapping == {0: 2, 1: 0, 2: 1}

    # Fallback: no headers -> use first row
    a2 = ParsedTable(
        rows=1,
        cols=2,
        headers=[],
        cells=[[ParsedCell(0, 0, "Col A"), ParsedCell(0, 1, "Col B")]],
    )
    b2 = ParsedTable(
        rows=1,
        cols=2,
        headers=[],
        cells=[[ParsedCell(0, 0, "Col A"), ParsedCell(0, 1, "Col B")]],
    )
    assert _detect_column_mapping(a2, b2) == {0: 0, 1: 1}
