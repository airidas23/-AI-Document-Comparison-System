from __future__ import annotations

import pytest

from comparison.models import Diff


def _diff(
    *,
    change_type: str = "content",
    diff_type: str = "modified",
    old: str | None = None,
    new: str | None = None,
    md: dict | None = None,
) -> Diff:
    return Diff(
        page_num=1,
        diff_type=diff_type,  # type: ignore[arg-type]
        change_type=change_type,  # type: ignore[arg-type]
        old_text=old,
        new_text=new,
        confidence=0.9,
        metadata=dict(md or {}),
    )


def test_diff_is_ocr_is_defensive_on_bad_metadata():
    from comparison.diff_classifier import _diff_is_ocr

    d = _diff(md={"is_ocr": True})
    assert _diff_is_ocr(d) is True

    class BadMeta:
        def get(self, *_a, **_k):
            raise RuntimeError("boom")

    d.metadata = BadMeta()  # type: ignore[assignment]
    assert _diff_is_ocr(d) is False


def test_classify_single_diff_respects_preclassified_non_content():
    from comparison.diff_classifier import _classify_single_diff

    d = _diff(change_type="layout", old=None, new=None, md={"layout_shift": True})
    out = _classify_single_diff(d)
    assert out.change_type == "layout"
    assert out.metadata.get("subtype") == "layout_shift"


def test_classify_single_diff_forces_content_for_text_level_subtypes():
    from comparison.diff_classifier import _classify_single_diff

    d = _diff(change_type="layout", old="a b", new="a  b", md={"subtype": "whitespace"})
    out = _classify_single_diff(d)
    assert out.change_type == "content"


@pytest.mark.parametrize(
    "md,diff_type,expected",
    [
        ({"table_change": "table_added", "table_structure": {"rows": 2, "cols": 3}}, "added", "Table added (2x3)"),
        ({"table_change": "table_deleted", "table_structure": {"rows": 1, "cols": 1}}, "deleted", "Table removed (1x1)"),
        (
            {"table_change": "structure", "old_structure": {"rows": 1, "cols": 1}, "new_structure": {"rows": 3, "cols": 2}},
            "modified",
            "Table structure changed",
        ),
        ({"table_change": "cell_content", "row": 4, "col": 5}, "modified", "Table cell changed (row 4, col 5)"),
    ],
)
def test_generate_description_table_branches(md, diff_type, expected):
    from comparison.diff_classifier import _generate_description

    d = _diff(change_type="layout", diff_type=diff_type, old=None, new=None, md=md)
    _generate_description(d)
    assert expected in d.metadata.get("description", "")


@pytest.mark.parametrize(
    "hf_change,diff_type,expected",
    [("header", "added", "Header added"), ("footer", "deleted", "Footer removed"), ("header", "modified", "Header changed")],
)
def test_generate_description_header_footer_branches(hf_change, diff_type, expected):
    from comparison.diff_classifier import _generate_description

    d = _diff(change_type="layout", diff_type=diff_type, old=None, new=None, md={"header_footer_change": hf_change})
    _generate_description(d)
    assert d.metadata.get("description") == expected


def test_generate_description_figure_numbering_branch():
    from comparison.diff_classifier import _generate_description

    d = _diff(
        change_type="visual",
        diff_type="modified",
        old=None,
        new=None,
        md={"figure_change": "numbering", "old_number": "Figure 1", "new_number": "Figure 2"},
    )
    _generate_description(d)
    assert "Figure numbering changed" in d.metadata.get("description", "")


def test_generate_description_text_added_preview_is_truncated():
    from comparison.diff_classifier import _generate_description

    text = "x" * 80
    d = _diff(change_type="content", diff_type="added", old=None, new=text, md={})
    _generate_description(d)
    desc = d.metadata.get("description", "")
    assert desc.startswith("Text added: '")
    assert desc.endswith("...'")


def test_generate_description_text_modified_preview_contains_arrow():
    from comparison.diff_classifier import _generate_description

    d = _diff(change_type="content", diff_type="modified", old="old text here", new="new text here", md={})
    _generate_description(d)
    assert "→" in d.metadata.get("description", "")
