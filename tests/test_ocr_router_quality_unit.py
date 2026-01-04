from __future__ import annotations

from types import SimpleNamespace

import pytest

from comparison.models import PageData, TextBlock


def _page(page_num: int, *, blocks: list[TextBlock] | None = None) -> PageData:
    return PageData(page_num=page_num, width=600.0, height=800.0, blocks=blocks or [], metadata={})


def _blk(text: str, *, conf: object | None = None) -> TextBlock:
    md = {}
    if conf is not None:
        md["confidence"] = conf
    return TextBlock(text=text, bbox={"x": 0, "y": 0, "width": 10, "height": 10}, metadata=md)


def test_get_engine_render_dpi_defaults_and_clamps(monkeypatch):
    from extraction.ocr_router import _get_engine_render_dpi
    from config.settings import settings

    monkeypatch.setattr(settings, "deepseek_render_dpi", 0, raising=False)
    monkeypatch.setattr(settings, "tesseract_render_dpi", 123, raising=False)
    monkeypatch.setattr(settings, "paddle_render_dpi", 200, raising=False)

    assert _get_engine_render_dpi("deepseek") == 150  # clamp
    assert _get_engine_render_dpi("tesseract") == 123
    assert _get_engine_render_dpi("paddle") == 200
    assert _get_engine_render_dpi(None) == 200


def test_annotate_ocr_quality_flags_low_confidence(monkeypatch):
    from extraction.ocr_router import _annotate_ocr_quality
    from config.settings import settings

    monkeypatch.setattr(settings, "ocr_quality_min_chars_per_page", 10, raising=False)
    monkeypatch.setattr(settings, "ocr_quality_min_avg_confidence", 0.8, raising=False)
    monkeypatch.setattr(settings, "ocr_quality_max_gibberish_ratio", 0.2, raising=False)

    p = _page(1, blocks=[_blk("ok text", conf=0.9), _blk("bad \ufffd\ufffd\ufffd", conf=0.1)])
    _annotate_ocr_quality([p], engine_name="tesseract")

    q = p.metadata["ocr_quality"]
    assert q["engine"] == "tesseract"
    assert q["char_count"] > 0
    assert q["low_confidence"] is True
    assert p.metadata["ocr_low_confidence"] is True


def test_is_ocr_successful_respects_low_quality_and_char_fallback(monkeypatch):
    from extraction.ocr_router import _is_ocr_successful
    from config.settings import settings

    monkeypatch.setattr(settings, "min_ocr_blocks_per_page", 2, raising=False)

    # Enough blocks but low-quality should not count.
    p1 = _page(1, blocks=[_blk("a"), _blk("b")])
    p1.metadata["ocr_quality"] = {"low_confidence": True}
    assert _is_ocr_successful([p1], engine_name="paddle") is False

    # Char fallback: enough total chars and not all pages low-quality.
    p2 = _page(1, blocks=[_blk("x" * 40)])
    p2.metadata["ocr_quality"] = {"low_confidence": False}
    assert _is_ocr_successful([p2], engine_name="paddle") is True

    # DeepSeek: fewer blocks required but higher char threshold.
    # Use zero blocks to avoid the early min_blocks success path.
    p3 = _page(1, blocks=[])
    p3.metadata["ocr_quality"] = {"low_confidence": False}
    assert _is_ocr_successful([p3], engine_name="deepseek") is False


def test_add_layout_metadata_merges_and_handles_failure(monkeypatch):
    from extraction.ocr_router import _add_layout_metadata

    pages = [_page(1, blocks=[_blk("x")]), _page(2, blocks=[_blk("y")])]

    # Success path
    def analyze_layout_ok(_path):
        p1 = _page(1)
        p1.metadata = {"tables": ["t"], "figures": ["f"], "text_blocks": ["tb"], "layout_method": "yolo"}
        return [p1]

    monkeypatch.setattr("extraction.layout_analyzer.analyze_layout", analyze_layout_ok)

    out = _add_layout_metadata(SimpleNamespace(), pages)
    assert out[0].metadata["layout_analyzed"] is True
    assert out[0].metadata["tables"] == ["t"]
    assert out[1].metadata["layout_analyzed"] is True
    assert out[1].metadata["tables"] == []

    # Failure path
    def analyze_layout_boom(_path):
        raise RuntimeError("nope")

    pages2 = [_page(1, blocks=[_blk("x")])]
    monkeypatch.setattr("extraction.layout_analyzer.analyze_layout", analyze_layout_boom)

    out2 = _add_layout_metadata(SimpleNamespace(), pages2)
    assert out2[0].metadata["layout_analyzed"] is False
    assert out2[0].metadata["tables"] == []
    assert "layout_error" in out2[0].metadata


def test_create_empty_pages_with_metadata_uses_fitz(monkeypatch, tmp_path):
    from extraction.ocr_router import _create_empty_pages_with_metadata

    # Provide a fake fitz module only for this test.
    class _FakeRect:
        width = 100.0
        height = 200.0

    class _FakePage:
        def __init__(self, number: int):
            self.number = number
            self.rect = _FakeRect()

    class _FakeDoc:
        def __iter__(self):
            return iter([_FakePage(0), _FakePage(1)])

        def close(self):
            return None

    fake_fitz = SimpleNamespace(open=lambda _path: _FakeDoc())
    monkeypatch.setitem(__import__("sys").modules, "fitz", fake_fitz)

    fake_pdf = tmp_path / "f.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n%fake\n")

    pages = _create_empty_pages_with_metadata(
        fake_pdf,
        attempted_engines=["tesseract"],
        last_error="boom",
        policy="auto",
        engine_selected="tesseract",
        attempts=[{"engine": "tesseract", "status": "failed"}],
        failure_reason="all_failed",
    )

    assert [p.page_num for p in pages] == [1, 2]
    assert pages[0].metadata["ocr_status"] == "failed"
    assert pages[0].metadata["ocr_engine_selected"] == "tesseract"
    assert pages[0].metadata["dpi"] > 0
