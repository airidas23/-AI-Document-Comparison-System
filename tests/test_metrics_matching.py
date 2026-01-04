import pytest

from comparison.models import Diff
from utils.metrics import calculate_change_detection_metrics


def test_match_allows_modified_vs_added_with_text_similarity() -> None:
    gt = [
        Diff(
            page_num=1,
            diff_type="added",
            change_type="content",
            old_text=None,
            new_text="Hello world",
            bbox=None,
            confidence=1.0,
        )
    ]

    pred = [
        Diff(
            page_num=1,
            diff_type="modified",
            change_type="content",
            old_text="",
            new_text="Hello, world!",
            bbox={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05},
            confidence=0.9,
        )
    ]

    m = calculate_change_detection_metrics(pred, gt)
    assert m.recall == 1.0
    assert m.precision == 1.0


def test_match_allows_modified_vs_deleted_using_old_text() -> None:
    gt = [
        Diff(
            page_num=2,
            diff_type="deleted",
            change_type="content",
            old_text="Removed sentence.",
            new_text=None,
            bbox=None,
            confidence=1.0,
        )
    ]

    pred = [
        Diff(
            page_num=2,
            diff_type="modified",
            change_type="content",
            old_text="Removed sentence",
            new_text="",
            bbox=None,
            confidence=0.9,
        )
    ]

    m = calculate_change_detection_metrics(pred, gt)
    assert m.recall == 1.0
    assert m.precision == 1.0


def test_one_to_one_matching_prevents_double_tp() -> None:
    gt = [
        Diff(
            page_num=1,
            diff_type="modified",
            change_type="content",
            old_text="Alpha",
            new_text="Beta",
            bbox=None,
            confidence=1.0,
        )
    ]

    pred = [
        Diff(
            page_num=1,
            diff_type="modified",
            change_type="content",
            old_text="Alpha",
            new_text="Beta",
            bbox=None,
            confidence=0.9,
        ),
        Diff(
            page_num=1,
            diff_type="modified",
            change_type="content",
            old_text="Alpha",
            new_text="Beta",
            bbox=None,
            confidence=0.8,
        ),
    ]

    m = calculate_change_detection_metrics(pred, gt)
    # Only one GT exists; second prediction must be FP.
    assert m.recall == 1.0
    assert m.precision == 0.5


def test_text_mismatch_does_not_match() -> None:
    gt = [
        Diff(
            page_num=1,
            diff_type="modified",
            change_type="content",
            old_text="Cats",
            new_text="Dogs",
            bbox=None,
            confidence=1.0,
        )
    ]

    pred = [
        Diff(
            page_num=1,
            diff_type="modified",
            change_type="content",
            old_text="Completely different",
            new_text="Nothing alike",
            bbox=None,
            confidence=0.9,
        )
    ]

    m = calculate_change_detection_metrics(pred, gt)
    assert m.recall == 0.0
    assert m.precision == 0.0
