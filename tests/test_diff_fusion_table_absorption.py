from comparison.diff_fusion import fuse_diffs
from comparison.models import Diff


def test_table_fusion_absorbs_table_like_noise_but_not_paragraph_diff() -> None:
    # A table structure diff (anchor)
    table = Diff(
        page_num=1,
        diff_type="modified",
        change_type="layout",
        old_text="Table (6x4)",
        new_text="Table (6x8)",
        bbox={"x": 0.10, "y": 0.47, "width": 0.32, "height": 0.15},
        confidence=0.85,
        metadata={"type": "table", "table_change": "structure", "is_ocr": True},
    )

    # Table-local OCR text noise that should be absorbed into the table cluster
    table_like_noise = Diff(
        page_num=1,
        diff_type="modified",
        change_type="content",
        old_text="| Metric | Original |",
        new_text="| Metric | Revised | Delta |",
        bbox={"x": 0.11, "y": 0.48, "width": 0.30, "height": 0.14},
        confidence=0.7,
        metadata={"type": "line", "is_ocr": True},
    )

    # A long paragraph diff overlapping the table region; must NOT be absorbed
    paragraph_typo = Diff(
        page_num=1,
        diff_type="modified",
        change_type="content",
        old_text=(
            "The evaluation framework uses these synthetic pairs to measure detection accuracy, "
            "false positive rates, and the ability to correctly classify different types of changes."
        ),
        new_text=(
            "The evaluation framework uses these synthetic pairs to measure detection accuracy, "
            "false positive raates, and the ability to correctly classify different types of changes."
        ),
        # Intentionally overlaps the table bbox in y-range and uses a wide bbox (like the reported bug)
        bbox={"x": 0.09, "y": 0.41, "width": 0.75, "height": 0.21},
        confidence=0.75,
        metadata={"type": "line", "is_ocr": True, "subtype": "character_change"},
    )

    fused = fuse_diffs(
        [("table", [table]), ("line", [table_like_noise, paragraph_typo])],
        strategy="triangulation",
        iou_threshold=0.3,
    )

    # Paragraph typo must survive as a non-table diff.
    typo_hits = [
        d
        for d in fused
        if (d.old_text and "rates" in d.old_text) or (d.new_text and "raates" in d.new_text)
    ]
    assert len(typo_hits) == 1
    assert (typo_hits[0].metadata or {}).get("type") != "table"

    # There should be exactly one table diff, and it should absorb the table-like noise.
    table_diffs = [d for d in fused if (d.metadata or {}).get("type") == "table"]
    assert len(table_diffs) == 1

    # Table diff should not expand to the paragraph-wide bbox.
    td_bbox = table_diffs[0].bbox or {}
    assert td_bbox.get("width", 0.0) < 0.6

    # Absorbed count should be present (at least 1) for the table-like noise.
    absorbed = (table_diffs[0].metadata or {}).get("absorbed_line_diffs", 0)
    assert absorbed >= 1
