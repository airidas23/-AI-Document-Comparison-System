"""Tests for comparison modules."""
from __future__ import annotations

from comparison.alignment import align_pages, align_sections
from comparison.diff_classifier import classify_diffs, get_diff_summary
from comparison.models import Diff, PageData, Style, TextBlock


def test_align_pages_basic():
    """Test basic page alignment."""
    pages_a = [
        PageData(page_num=1, width=612, height=792),
        PageData(page_num=2, width=612, height=792),
    ]
    pages_b = [PageData(page_num=1, width=612, height=792)]
    mapping = align_pages(pages_a, pages_b)
    assert mapping[1][0] == 1
    assert mapping[2][0] == 1


def test_align_pages_same_length():
    """Test alignment when documents have same page count."""
    pages_a = [
        PageData(page_num=1, width=612, height=792),
        PageData(page_num=2, width=612, height=792),
    ]
    pages_b = [
        PageData(page_num=1, width=612, height=792),
        PageData(page_num=2, width=612, height=792),
    ]
    mapping = align_pages(pages_a, pages_b)
    assert len(mapping) == 2
    assert mapping[1][1] == 1.0  # Full confidence for same-length docs


def test_align_sections():
    """Test block-level alignment."""
    page_a = PageData(
        page_num=1,
        width=612,
        height=792,
        blocks=[
            TextBlock(text="Block 1", bbox={"x": 0, "y": 0, "width": 100, "height": 20}),
            TextBlock(text="Block 2", bbox={"x": 0, "y": 30, "width": 100, "height": 20}),
        ],
    )
    page_b = PageData(
        page_num=1,
        width=612,
        height=792,
        blocks=[
            TextBlock(text="Block 1", bbox={"x": 0, "y": 0, "width": 100, "height": 20}),
            TextBlock(text="Block 2 modified", bbox={"x": 0, "y": 30, "width": 100, "height": 20}),
        ],
    )
    alignment = align_sections(page_a, page_b)
    assert len(alignment) == 2
    assert 0 in alignment
    assert 1 in alignment


def test_classify_diffs():
    """Test diff classification."""
    diffs = [
        Diff(
            page_num=1,
            diff_type="modified",
            change_type="content",
            old_text="Hello world",
            new_text="Hello world!",
            confidence=0.8,
        ),
        Diff(
            page_num=1,
            diff_type="modified",
            change_type="content",
            old_text="Test",
            new_text="TEST",
            confidence=0.7,
        ),
    ]
    classified = classify_diffs(diffs)
    assert len(classified) == 2
    # First should be punctuation change
    assert classified[0].change_type == "formatting"
    assert classified[0].metadata.get("subtype") == "punctuation"
    # Second should be case change
    assert classified[1].change_type == "formatting"
    assert classified[1].metadata.get("subtype") == "case"


def test_get_diff_summary():
    """Test diff summary generation."""
    diffs = [
        Diff(
            page_num=1,
            diff_type="added",
            change_type="content",
            old_text=None,
            new_text="New text",
            confidence=0.9,
        ),
        Diff(
            page_num=1,
            diff_type="deleted",
            change_type="content",
            old_text="Old text",
            new_text=None,
            confidence=0.9,
        ),
        Diff(
            page_num=2,
            diff_type="modified",
            change_type="formatting",
            old_text="Text",
            new_text="Text",
            confidence=0.8,
        ),
    ]
    summary = get_diff_summary(diffs)
    assert summary["total"] == 3
    assert summary["by_type"]["added"] == 1
    assert summary["by_type"]["deleted"] == 1
    assert summary["by_type"]["modified"] == 1
    assert summary["by_change_type"]["content"] == 2
    assert summary["by_change_type"]["formatting"] == 1
