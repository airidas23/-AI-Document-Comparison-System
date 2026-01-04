import pytest

from comparison.hierarchical_alignment import (
    _align_headings,
    _get_section_segments,
    align_with_dtw,
    align_with_needleman_wunsch,
    hierarchical_align,
    segment_document,
)
from comparison.models import PageData, TextBlock


def _make_page(block_texts, *, headings=None, tables=None, lists=None):
    blocks = [
        TextBlock(text=t, bbox={"x": 10, "y": 10 + i * 15, "width": 500, "height": 12}, metadata={})
        for i, t in enumerate(block_texts)
    ]
    page = PageData(page_num=1, width=595, height=842, blocks=blocks)
    page.metadata = {
        "has_markdown_structure": True,
        "headings": headings or [],
        "tables": tables or [],
        "lists": lists or [],
    }
    return page


def test_segment_document_builds_hierarchy_for_headings_and_paragraphs():
    page = _make_page(
        ["INTRO", "First paragraph line.", "Second line."],
        headings=[{"index": 0, "level": 1, "text": "INTRO"}],
    )

    segments = segment_document(page)
    assert [s.block_type for s in segments] == ["heading", "paragraph"]

    heading, paragraph = segments
    assert heading.heading_level == 1
    assert paragraph.parent_index == 0
    assert 1 in heading.children_indices


def test_segment_document_groups_table_rows_into_single_table_segment():
    page = _make_page(
        ["H1", "A|B", "1|2", "Tail"],
        headings=[{"index": 0, "level": 1, "text": "H1"}],
        tables=[{"rows": [{"index": 1}, {"index": 2}]}],
    )

    segments = segment_document(page)
    types = [s.block_type for s in segments]
    assert types == ["heading", "table", "paragraph"]
    assert "A|B" in segments[1].text
    assert "1|2" in segments[1].text


def test_align_headings_prefers_level_and_text_overlap():
    page_a = _make_page(["1. Results"], headings=[{"index": 0, "level": 1, "text": "Results"}])
    page_b = _make_page(["1. Results"], headings=[{"index": 0, "level": 1, "text": "Results"}])

    segs_a = segment_document(page_a)
    segs_b = segment_document(page_b)

    headings_a = [(i, s) for i, s in enumerate(segs_a) if s.block_type == "heading"]
    headings_b = [(i, s) for i, s in enumerate(segs_b) if s.block_type == "heading"]

    mapping = _align_headings(headings_a, headings_b)
    assert mapping

    (matched_idx_b, conf) = mapping[0]
    assert matched_idx_b == 0
    assert conf >= 0.7


def test_get_section_segments_includes_children_until_next_heading():
    page = _make_page(
        ["H1", "Para 1", "H2", "Para 2"],
        headings=[
            {"index": 0, "level": 1, "text": "H1"},
            {"index": 2, "level": 1, "text": "H2"},
        ],
    )
    segments = segment_document(page)

    # segments: [heading(H1), paragraph(Para 1), heading(H2), paragraph(Para 2)]
    section = _get_section_segments(segments, 0)
    section_indices = [idx for idx, _ in section]
    assert 1 in section_indices
    assert 2 not in section_indices


@pytest.mark.parametrize("aligner", [align_with_dtw, align_with_needleman_wunsch])
def test_paragraph_alignment_returns_reasonable_mapping(aligner):
    seg_a = [(1, type("S", (), {"text": "alpha beta", "block_type": "paragraph"})())]
    seg_b = [(5, type("S", (), {"text": "alpha beta", "block_type": "paragraph"})())]

    mapping = aligner(seg_a, seg_b)
    assert mapping
    assert mapping[1][0] == 5
    assert mapping[1][1] >= 0.5


def test_hierarchical_align_maps_paragraphs_under_matched_heading():
    page_a = _make_page(
        ["INTRO", "Hello world", "Second sentence"],
        headings=[{"index": 0, "level": 1, "text": "INTRO"}],
    )
    page_b = _make_page(
        ["INTRO", "Hello world!", "Second sentence"],
        headings=[{"index": 0, "level": 1, "text": "INTRO"}],
    )

    mapping = hierarchical_align(page_a, page_b, use_dtw=True)
    assert mapping

    # At least the paragraph segment (index 1) should align.
    assert 1 in mapping
    assert mapping[1][0] == 1
    assert mapping[1][1] > 0.3


def test_hierarchical_align_empty_pages_returns_empty_mapping():
    page_a = PageData(page_num=1, width=595, height=842, blocks=[])
    page_a.metadata = {}
    page_b = PageData(page_num=1, width=595, height=842, blocks=[])
    page_b.metadata = {}

    assert hierarchical_align(page_a, page_b) == {}
