import pytest

from comparison.diff_classifier import classify_diffs
from comparison.models import Diff


def test_ocr_noise_filter_keeps_single_char_typo_even_without_metadata_counts():
    """OCR aggressive noise filter should not drop real 1-char typos when producer didn't set ocr_changed_chars.

    Regression: TextComparator can emit a character-change diff without populating
    metadata['ocr_changed_chars']/['ocr_change_ratio']. The noise filter used to
    treat this as 0-char noise and remove it.
    """

    diffs = [
        Diff(
            page_num=1,
            diff_type="modified",
            change_type="content",
            old_text="rates,",
            new_text="raates,",
            bbox={"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.05},
            confidence=0.5,
            metadata={"is_ocr": True},
        )
    ]

    classified = classify_diffs(diffs)

    # The typo should survive filtering and remain classified.
    assert len(classified) == 1
    assert classified[0].old_text == "rates,"
    assert classified[0].new_text == "raates,"

    # And it should now have computed OCR change metrics cached.
    md = classified[0].metadata or {}
    assert md.get("ocr_changed_chars", 0) >= 1
    assert md.get("ocr_change_ratio", 0.0) > 0.0
