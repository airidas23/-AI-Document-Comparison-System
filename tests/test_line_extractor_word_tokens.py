from __future__ import annotations

from extraction.line_extractor import extract_lines_from_pages
from comparison.models import PageData, TextBlock


def test_extract_lines_from_pages_uses_word_metadata_tokens():
    page = PageData(
        page_num=1,
        width=600.0,
        height=800.0,
        blocks=[
            TextBlock(
                text="false positive rates",
                bbox={"x": 100.0, "y": 200.0, "width": 300.0, "height": 20.0},
                metadata={
                    "confidence": 0.9,
                    "words": [
                        {"text": "false", "bbox": {"x": 100.0, "y": 200.0, "width": 60.0, "height": 20.0}, "conf": 0.95},
                        {"text": "positive", "bbox": {"x": 165.0, "y": 200.0, "width": 85.0, "height": 20.0}, "conf": 0.92},
                        {"text": "rates", "bbox": {"x": 255.0, "y": 200.0, "width": 55.0, "height": 20.0}, "conf": 0.91},
                    ],
                },
            )
        ],
    )

    out = extract_lines_from_pages([page])
    assert len(out) == 1
    assert out[0].lines

    # All word tokens should be preserved
    line = out[0].lines[0]
    assert [t.text for t in line.tokens] == ["false", "positive", "rates"]
    assert line.tokens[2].bbox["x"] == 255.0
    assert line.tokens[2].bbox["width"] == 55.0


def test_extract_lines_from_pages_fallback_approximates_words_when_missing_metadata():
    page = PageData(
        page_num=1,
        width=600.0,
        height=800.0,
        blocks=[
            TextBlock(
                text="false positive rates",
                bbox={"x": 100.0, "y": 200.0, "width": 300.0, "height": 20.0},
                metadata={"confidence": 0.8},
            )
        ],
    )

    out = extract_lines_from_pages([page])
    assert out[0].lines
    line = out[0].lines[0]

    # Fallback creates 3 whitespace-delimited tokens
    assert [t.text for t in line.tokens] == ["false", "positive", "rates"]
    # Bboxes should be within the original block bbox bounds
    xs = [t.bbox["x"] for t in line.tokens]
    assert min(xs) >= 100.0
    assert max(xs) <= 400.0
