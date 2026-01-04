"""Regression tests for OCR paragraph merging behavior."""

from __future__ import annotations

from comparison.line_comparison import _merge_lines_to_paragraphs
from comparison.models import Line, PageData


def test_merge_lines_to_paragraphs_does_not_cross_table_boundary():
    page = PageData(page_num=1, width=600, height=800)
    # One detected table region in the middle of the page.
    page.metadata = {
        "tables": [
            {
                "bbox": {
                    "x": 50,
                    "y": 140,
                    "width": 500,
                    "height": 200,
                }
            }
        ]
    }

    # Two normal paragraph lines above the table.
    l1 = Line(
        line_id="l1",
        bbox={"x": 60, "y": 100, "width": 300, "height": 10},
        text="This is a paragraph line",
        confidence=0.9,
        reading_order=0,
        tokens=[],
    )
    l2 = Line(
        line_id="l2",
        bbox={"x": 60, "y": 115, "width": 320, "height": 10},
        text="continuation of the paragraph",
        confidence=0.9,
        reading_order=1,
        tokens=[],
    )

    # A table line close enough in Y that it would normally be merged by the gap heuristic.
    tl = Line(
        line_id="t1",
        bbox={"x": 80, "y": 145, "width": 420, "height": 10},
        text="Metric Delta",
        confidence=0.8,
        reading_order=2,
        tokens=[],
    )

    merged = _merge_lines_to_paragraphs([l1, l2, tl], page=page)

    # Expect two paragraphs: one above-table paragraph (l1+l2) and one table paragraph (tl).
    assert len(merged) == 2
    assert "Metric Delta" not in merged[0].text
    assert merged[1].text == "Metric Delta"
