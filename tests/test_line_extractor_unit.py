from __future__ import annotations

from pathlib import Path

import pytest

from comparison.models import PageData, TextBlock, Token


def test_normalize_text_for_id_and_quantize():
    from extraction.line_extractor import normalize_text_for_id, quantize

    assert normalize_text_for_id("Hello,   WORLD!!!") == "hello world"
    assert normalize_text_for_id("") == ""

    assert quantize(0.11, bin_size=0.1) == 0.1
    assert quantize(0.11, bin_size=0.0) == 0.11


def test_bbox_union_and_make_line_id_stability():
    from extraction.line_extractor import bbox_union, make_line_id

    assert bbox_union([]) == {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}

    b = bbox_union(
        [
            {"x": 10, "y": 10, "width": 5, "height": 5},
            {"x": 12, "y": 8, "width": 10, "height": 2},
        ]
    )
    assert b["x"] == 10
    assert b["y"] == 8
    assert b["width"] > 0
    assert b["height"] > 0

    line_id1 = make_line_id("Some text", b, page_num=1, page_width=200, page_height=200)
    line_id2 = make_line_id("Some text", b, page_num=1, page_width=200, page_height=200)
    assert line_id1 == line_id2

    line_id3 = make_line_id("Some text", b, page_num=2, page_width=200, page_height=200)
    assert line_id3 != line_id1


def test_group_tokens_into_lines_and_build_lines():
    from extraction.line_extractor import _build_lines_from_token_groups, group_tokens_into_lines

    tokens = [
        Token(token_id="t1", bbox={"x": 10, "y": 10, "width": 10, "height": 10}, text="Hello"),
        Token(token_id="t2", bbox={"x": 30, "y": 12, "width": 10, "height": 10}, text="world"),
        Token(token_id="t3", bbox={"x": 10, "y": 80, "width": 10, "height": 10}, text="Next"),
    ]

    groups = group_tokens_into_lines(tokens, y_threshold=5.0)
    assert len(groups) == 2

    lines = _build_lines_from_token_groups(groups, page_num=1, page_width=200, page_height=200)
    assert [ln.text for ln in lines] == ["Hello world", "Next"]
    assert lines[0].reading_order == 0
    assert lines[1].reading_order == 1


def test_paddle_results_to_tokens_filters_and_scales(monkeypatch):
    from extraction.line_extractor import _paddle_results_to_tokens
    from config.settings import settings

    monkeypatch.setattr(settings, "paddle_text_rec_score_thresh", 0.8, raising=False)

    # One good token, one below threshold, one missing polygon
    ocr_result = [
        {
            "rec_texts": ["Good", "Bad"],
            "rec_scores": [0.95, 0.1],
            "dt_polys": [
                [(0, 0), (10, 0), (10, 10), (0, 10)],
                [(20, 0), (30, 0), (30, 10), (20, 10)],
            ],
        },
        {"rec_texts": ["NoPoly"], "rec_scores": [0.99], "dt_polys": []},
    ]

    tokens = _paddle_results_to_tokens(ocr_result, dpi=200, page_num=3)
    assert len(tokens) == 1
    assert tokens[0].text == "Good"
    assert tokens[0].token_id.startswith("p3_ocr")
    assert tokens[0].bbox["width"] > 0


def test_extract_lines_from_pages_word_metadata_and_fallbacks():
    from extraction.line_extractor import extract_lines_from_pages

    pages = [
        PageData(
            page_num=1,
            width=200.0,
            height=200.0,
            blocks=[
                # Word-level metadata path
                TextBlock(
                    text="ignored",
                    bbox={"x": 0, "y": 0, "width": 100, "height": 10},
                    metadata={
                        "confidence": 0.7,
                        "words": [
                            {"text": "A", "bbox": {"x": 0, "y": 0, "width": 10, "height": 10}, "conf": 0.9},
                            {"text": "B", "bbox": {"x": 20, "y": 0, "width": 10, "height": 10}},
                        ],
                    },
                ),
                # Approx-word token fallback path
                TextBlock(
                    text="Hello world here",
                    bbox={"x": 0, "y": 20, "width": 100, "height": 10},
                    metadata={"confidence": 1.0},
                ),
                # Block-level token path
                TextBlock(
                    text="Single",
                    bbox={"x": 0, "y": 40, "width": 100, "height": 10},
                    metadata={"confidence": 1.0},
                ),
            ],
            metadata={"ocr_engine_used": "paddle"},
        )
    ]

    out = extract_lines_from_pages(pages)
    assert out[0].lines
    assert out[0].metadata["line_extraction_method"] == "from_blocks"

    texts = [ln.text for ln in out[0].lines]
    assert any("A B" in t for t in texts)
    assert any("Hello world here" in t for t in texts)
    assert any("Single" in t for t in texts)


def test_extract_lines_prefers_existing_blocks(monkeypatch):
    from extraction.line_extractor import extract_lines

    pages = [PageData(page_num=1, width=100.0, height=100.0, blocks=[TextBlock(text="x", bbox={"x": 0, "y": 0, "width": 10, "height": 10})])]

    called = {"n": 0}

    def fake_extract_lines_from_pages(p):
        called["n"] += 1
        return p

    monkeypatch.setattr("extraction.line_extractor.extract_lines_from_pages", fake_extract_lines_from_pages)

    out = extract_lines(pages)
    assert out is pages
    assert called["n"] == 1


def test_extract_document_lines_fallbacks(monkeypatch, tmp_path: Path):
    from extraction.line_extractor import extract_document_lines

    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")

    # Digital returns empty lines -> OCR fallback
    monkeypatch.setattr("extraction.line_extractor.extract_digital_lines", lambda p: [PageData(page_num=1, width=1, height=1, lines=[])])
    monkeypatch.setattr("extraction.line_extractor.extract_ocr_lines", lambda p: [PageData(page_num=1, width=1, height=1, lines=[])])

    out = extract_document_lines(pdf)
    assert len(out) == 1

    # Digital throws -> OCR used
    def boom(_):
        raise RuntimeError("nope")

    monkeypatch.setattr("extraction.line_extractor.extract_digital_lines", boom)
    monkeypatch.setattr("extraction.line_extractor.extract_ocr_lines", lambda p: [])
    assert extract_document_lines(pdf) == []
