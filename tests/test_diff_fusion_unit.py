from __future__ import annotations


from comparison.models import Diff


def _d(
    *,
    page: int = 1,
    diff_type: str = "modified",
    change_type: str = "content",
    old: str | None = "a",
    new: str | None = "b",
    bbox: dict | None = None,
    conf: float = 0.8,
    md: dict | None = None,
    bbox_b: dict | None = None,
) -> Diff:
    return Diff(
        page_num=page,
        diff_type=diff_type,  # type: ignore[arg-type]
        change_type=change_type,  # type: ignore[arg-type]
        old_text=old,
        new_text=new,
        bbox=bbox,
        bbox_b=bbox_b,
        confidence=conf,
        metadata=dict(md or {}),
    )


def test_iou_and_overlap_helpers():
    from comparison.diff_fusion import calculate_iou, diffs_overlap

    assert calculate_iou(None, None) == 0.0
    assert calculate_iou({"x": 0, "y": 0, "width": 0.1, "height": 0.1}, None) == 0.0

    a = {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}
    b = {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}
    c = {"x": 0.8, "y": 0.8, "width": 0.1, "height": 0.1}

    assert calculate_iou(a, b) == 1.0
    assert calculate_iou(a, c) == 0.0

    # text-overlap path (no bbox, same page)
    d1 = _d(page=1, old="This is a sufficiently long shared snippet", new=None, bbox=None)
    d2 = _d(page=1, old="shared snippet", new=None, bbox=None)
    assert diffs_overlap(d1, d2) is True

    # page mismatch always false
    d3 = _d(page=2, old="This is a sufficiently long shared snippet", new=None, bbox=None)
    assert diffs_overlap(d1, d3) is False


def test_diffcluster_bbox_b_prefers_tightest_candidate():
    from comparison.diff_fusion import DiffCluster

    wide_b = {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8}
    tight_b = {"x": 0.2, "y": 0.2, "width": 0.1, "height": 0.1}

    d1 = _d(bbox={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}, bbox_b=wide_b)
    d2 = _d(bbox={"x": 0.2, "y": 0.2, "width": 0.1, "height": 0.1}, md={"bbox_b": tight_b})

    cluster = DiffCluster(diffs=[d1, d2], modules=["m1", "m2"])
    assert cluster.merged_bbox is not None
    assert cluster.merged_bbox_b == tight_b


def test_consensus_change_type_special_cases():
    from comparison.diff_fusion import determine_consensus_change_type

    # Table structure should stay layout unless it is a semantic column change.
    table_structure = _d(
        change_type="layout",
        md={"type": "table", "table_change": "structure"},
    )
    noise_content = _d(change_type="content", md={"type": "line"})
    assert determine_consensus_change_type([table_structure, noise_content]) == "layout"

    table_cols = _d(
        change_type="layout",
        md={"type": "table", "table_change": "structure", "subtype": "table_columns_changed"},
    )
    assert determine_consensus_change_type([table_cols, noise_content]) == "content"

    figure_num = _d(
        change_type="content",
        md={"type": "figure", "figure_change": "numbering"},
    )
    assert determine_consensus_change_type([figure_num, noise_content]) == "visual"


def test_merge_metadata_table_priority_counts_absorbed_and_sets_fusion_fields():
    from comparison.diff_fusion import merge_metadata

    table = _d(md={"type": "table", "table_change": "structure", "description": "Table structure changed"})
    line1 = _d(md={"type": "line", "subtype": "punctuation"})
    line2 = _d(md={"type": "line", "subtype": "whitespace"})

    md = merge_metadata([table, line1, line2], ["table", "line"])
    assert md.get("type") == "table"
    assert md.get("table_change") == "structure"
    assert md.get("absorbed_line_diffs") == 2
    assert md.get("description") == "Table structure changed"

    assert md.get("fusion_modules") == ["table", "line"]
    assert md.get("fusion_count") == 2
    assert md.get("fusion_diff_count") == 3


def test_calculate_fused_confidence_branches():
    from comparison.diff_fusion import DiffCluster, calculate_fused_confidence

    d = _d(conf=0.9)
    assert calculate_fused_confidence(DiffCluster(diffs=[d], modules=["a", "b", "c"]), 3) == 0.95
    assert calculate_fused_confidence(DiffCluster(diffs=[d], modules=["a", "b"]), 3) == 0.85

    out = calculate_fused_confidence(DiffCluster(diffs=[_d(conf=1.0)], modules=["a"]), 3)
    assert 0.3 <= out <= 0.75


def test_fuse_diffs_intersection_filter_and_formatting_aggregation():
    from comparison.diff_fusion import fuse_diffs

    # Intersection drops single-module clusters.
    only_one = _d(page=1, change_type="content", old="a", new="b", bbox={"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1})
    out = fuse_diffs([("m1", [only_one])], strategy="intersection")
    assert out == []

    # Repeating formatting diffs across 3 pages should aggregate to a single document-level diff.
    fmt = []
    for page in (1, 2, 3):
        fmt.append(
            _d(
                page=page,
                change_type="formatting",
                old="",
                new="",
                bbox={"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.02},
                md={"formatting_type": "font_size", "old_size": 10.0, "new_size": 12.0},
            )
        )

    fused = fuse_diffs([("fmt", fmt)], strategy="triangulation")
    assert len(fused) == 1
    assert fused[0].metadata.get("aggregated") is True
    assert fused[0].metadata.get("scope") == "document"
    assert fused[0].metadata.get("pages") == [1, 2, 3]
