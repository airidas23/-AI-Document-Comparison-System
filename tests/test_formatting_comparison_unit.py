from __future__ import annotations

from comparison.models import PageData, Style, TextBlock


def _blk(
    text: str,
    *,
    x: float,
    y: float,
    w: float = 100,
    h: float = 20,
    style: Style | None = None,
    words: list[dict] | None = None,
) -> TextBlock:
    md: dict = {}
    if words is not None:
        md["words"] = words
    return TextBlock(text=text, bbox={"x": x, "y": y, "width": w, "height": h}, style=style, metadata=md)


def _page(
    page_num: int,
    *,
    width: float = 600.0,
    height: float = 800.0,
    blocks: list[TextBlock],
    metadata: dict | None = None,
) -> PageData:
    return PageData(page_num=page_num, width=width, height=height, blocks=blocks, metadata=metadata or {})


def test_texts_similar_enough_for_formatting_basic():
    from comparison.formatting_comparison import _texts_similar_enough_for_formatting

    assert _texts_similar_enough_for_formatting("", "x") is False
    assert _texts_similar_enough_for_formatting("abc", "") is False

    # Containment with decent coverage
    assert _texts_similar_enough_for_formatting("hello world", "hello world!") is True

    # High similarity ratio (minor punctuation change)
    assert _texts_similar_enough_for_formatting("This is a test", "This is a test.") is True


def test_detect_any_ocr_looks_at_multiple_metadata_fields():
    from comparison.formatting_comparison import _detect_any_ocr

    p1 = _page(1, blocks=[], metadata={"extraction_method": "native"})
    p2 = _page(2, blocks=[], metadata={"line_extraction_method": "OCR_lines"})
    assert _detect_any_ocr([p1], [p2]) is True


def test_compare_formatting_skips_entirely_for_ocr_when_configured(monkeypatch):
    from comparison.formatting_comparison import compare_formatting
    from config.settings import settings

    monkeypatch.setattr(settings, "skip_formatting_for_ocr", True, raising=False)

    pages_a = [_page(1, blocks=[], metadata={"ocr_engine_used": "tesseract"})]
    pages_b = [_page(1, blocks=[], metadata={})]

    assert compare_formatting(pages_a, pages_b) == []


def test_compare_formatting_emits_block_style_and_page_layout_diffs(monkeypatch):
    from comparison import formatting_comparison
    from config.settings import settings

    monkeypatch.setattr(settings, "skip_formatting_for_ocr", False, raising=False)
    monkeypatch.setattr(settings, "formatting_change_threshold", 0.02, raising=False)
    monkeypatch.setattr(settings, "font_size_change_threshold_pt", 1.0, raising=False)
    monkeypatch.setattr(settings, "color_difference_threshold", 1, raising=False)

    # Deterministic alignment
    monkeypatch.setattr(formatting_comparison, "align_pages", lambda a, b, use_similarity=False: {1: (1, 0.9)})
    monkeypatch.setattr(formatting_comparison, "align_sections", lambda pa, pb: {0: 0})

    # Word-level metadata (same token) to exercise _compare_word_styles
    words_a = [
        {
            "text": "Hello",
            "bbox": {"x": 10, "y": 10, "width": 30, "height": 10},
            "style": {"font": "Arial", "size": 12, "bold": False, "italic": False, "color": [0, 0, 0]},
        }
    ]
    words_b = [
        {
            "text": "Hello",
            "bbox": {"x": 10, "y": 10, "width": 30, "height": 10},
            "style": {"font": "Times", "size": 14, "bold": True, "italic": True, "color": [10, 0, 0]},
        }
    ]

    block_a = _blk(
        "Hello",
        x=10,
        y=10,
        style=Style(font="Arial", size=12, bold=False, italic=False, color=(0, 0, 0)),
        words=words_a,
    )
    block_b = _blk(
        "Hello",
        x=10,
        y=10,
        style=Style(font="Times", size=14, bold=True, italic=True, color=(10, 0, 0)),
        words=words_b,
    )

    # Add a 2nd block to trigger spacing calculation
    page_a = _page(1, width=600, height=800, blocks=[block_a, _blk("X", x=10, y=100)])
    page_b = _page(1, width=630, height=800, blocks=[block_b, _blk("X", x=10, y=150)])

    diffs = formatting_comparison.compare_formatting([page_a], [page_b])

    kinds = {(d.change_type, d.metadata.get("formatting_type"), d.metadata.get("scope")) for d in diffs}

    # Block-level style diffs
    assert ("formatting", "font", None) in kinds
    assert ("formatting", "font_size", None) in kinds
    assert ("formatting", "style", None) in kinds
    assert ("formatting", "color", None) in kinds

    # Word-level diffs should still be present for font/style/color (font_size may be suppressed)
    assert ("formatting", "font", "word") in kinds
    assert ("formatting", "style", "word") in kinds
    assert ("formatting", "color", "word") in kinds

    # Page layout diffs (page size + spacing)
    assert ("layout", "page_size", None) in kinds
    assert ("layout", "spacing", None) in kinds


def test_compare_word_styles_can_report_font_size_when_not_suppressed(monkeypatch):
    from comparison.formatting_comparison import _compare_styles
    from config.settings import settings

    monkeypatch.setattr(settings, "skip_formatting_for_ocr", False, raising=False)
    monkeypatch.setattr(settings, "font_size_change_threshold_pt", 1.0, raising=False)
    monkeypatch.setattr(settings, "color_difference_threshold", 1000, raising=False)

    # No block-level size (so suppress_font_size=False and block font-size won't be reported)
    block_a = _blk(
        "Hello",
        x=10,
        y=10,
        style=Style(font="Arial", size=None),
        words=[
            {
                "text": "Hello",
                "bbox": {"x": 10, "y": 10, "width": 30, "height": 10},
                "style": {"font": "Arial", "size": 10, "bold": False, "italic": False, "color": [0, 0, 0]},
            }
        ],
    )
    block_b = _blk(
        "Hello",
        x=10,
        y=10,
        style=Style(font="Arial", size=None),
        words=[
            {
                "text": "Hello",
                "bbox": {"x": 10, "y": 10, "width": 30, "height": 10},
                "style": {"font": "Arial", "size": 14, "bold": False, "italic": False, "color": [0, 0, 0]},
            }
        ],
    )

    diffs = _compare_styles(block_a, block_b, page_num=1, confidence=1.0, page_width=600, page_height=800)
    assert any(d.metadata.get("formatting_type") == "font_size" and d.metadata.get("scope") == "word" for d in diffs)
