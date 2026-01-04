from __future__ import annotations

from PIL import Image
import types
from pathlib import Path
import sys
import pytest

from comparison.models import PageData, TextBlock


def test_maybe_denormalize_bbox_dict_normalized_to_absolute():
    from extraction.ocr_visualizer import _maybe_denormalize_bbox_dict

    bbox_norm = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}
    out = _maybe_denormalize_bbox_dict(bbox_norm, page_width=200.0, page_height=100.0)

    # Should be converted to absolute coords (not kept near 0..1 range).
    assert out["x"] > 1.0
    assert out["y"] > 1.0
    assert out["width"] > 1.0
    assert out["height"] > 1.0


def test_maybe_denormalize_bbox_dict_invalid_input_passthrough():
    from extraction.ocr_visualizer import _maybe_denormalize_bbox_dict

    bbox = {"x": "bad"}
    assert _maybe_denormalize_bbox_dict(bbox, page_width=100.0, page_height=100.0) is bbox
    assert _maybe_denormalize_bbox_dict({"x": 10, "y": 10, "width": 5, "height": 5}, 0, 0)["x"] == 10


def test_draw_ocr_bboxes_smoke():
    from extraction.ocr_visualizer import draw_ocr_bboxes

    img = Image.new("RGB", (200, 100), color=(255, 255, 255))

    page = PageData(
        page_num=1,
        width=200.0,
        height=100.0,
        blocks=[
            TextBlock(
                text="Hello",
                bbox={"x": 10, "y": 10, "width": 50, "height": 20},
                metadata={"confidence": 0.9},
            ),
            # Include a normalized bbox in metadata list to hit denormalization branch.
            TextBlock(
                text="World",
                bbox={"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.2},
                metadata={"bboxes": [{"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.2}]},
            ),
        ],
    )

    out = draw_ocr_bboxes(img, page, show_text=True, show_confidence=True)
    assert out is not img
    assert out.size == img.size

    # Basic signal that something was drawn (some pixel differs from white).
    pixels = out.getdata()
    assert any(p != (255, 255, 255) for p in list(pixels)[0:2000])


def test_visualize_ocr_on_pdf_page_with_stubs(monkeypatch, tmp_path: Path):
    from extraction.ocr_visualizer import visualize_ocr_on_pdf_page

    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

    class _FakePix:
        width = 10
        height = 10
        samples = b"\xff" * (10 * 10 * 3)

    class _FakePage:
        def get_pixmap(self, dpi=300):
            assert dpi == 300
            return _FakePix()

    class _FakeDoc:
        def __getitem__(self, idx):
            assert idx == 0
            return _FakePage()

        def close(self):
            return None

    monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda p: _FakeDoc()))

    # Stub paddle_ocr_engine.ocr_pdf to return PageData with blocks.
    from comparison.models import PageData, TextBlock

    fake_pages = [
        PageData(
            page_num=1,
            width=200.0,
            height=100.0,
            blocks=[TextBlock(text="Hi", bbox={"x": 10, "y": 10, "width": 10, "height": 10})],
            metadata={"ocr_engine_used": "paddle"},
        )
    ]

    monkeypatch.setitem(
        sys.modules,
        "extraction.paddle_ocr_engine",
        types.SimpleNamespace(ocr_pdf=lambda p: fake_pages),
    )

    out_img = visualize_ocr_on_pdf_page(str(pdf_path), page_num=0, ocr_engine="paddle")
    assert out_img.size == (10, 10)


def test_visualize_ocr_on_pdf_page_unknown_engine_raises(monkeypatch, tmp_path: Path):
    from extraction.ocr_visualizer import visualize_ocr_on_pdf_page

    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

    # Stub fitz open/render so we reach the engine switch.
    class _FakePix:
        width = 1
        height = 1
        samples = b"\xff\xff\xff"

    class _FakePage:
        def get_pixmap(self, dpi=300):
            return _FakePix()

    class _FakeDoc:
        def __getitem__(self, idx):
            return _FakePage()

        def close(self):
            return None

    monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda p: _FakeDoc()))

    with pytest.raises(ValueError):
        visualize_ocr_on_pdf_page(str(pdf_path), ocr_engine="nope")


def test_filter_contained_blocks_removes_wrapper_and_duplicate():
    from extraction.ocr_visualizer import _filter_contained_blocks

    wrapper = {"bbox": {"x": 0, "y": 0, "width": 100, "height": 100}}
    inner = {"bbox": {"x": 10, "y": 10, "width": 80, "height": 80}}
    inner2 = {"bbox": {"x": 11, "y": 11, "width": 79, "height": 79}}

    out = _filter_contained_blocks([wrapper, inner, inner2])
    assert len(out) == 1


def test_draw_ocr_bboxes_font_fallback(monkeypatch):
    from extraction.ocr_visualizer import draw_ocr_bboxes
    from PIL import ImageFont

    orig_truetype = ImageFont.truetype

    def _truetype_fail_only_for_helvetica(font, *args, **kwargs):
        if isinstance(font, str) and font.endswith("/System/Library/Fonts/Helvetica.ttc"):
            raise OSError("no font")
        return orig_truetype(font, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", _truetype_fail_only_for_helvetica)

    img = Image.new("RGB", (50, 50), color=(255, 255, 255))
    page = PageData(
        page_num=1,
        width=50.0,
        height=50.0,
        blocks=[
            TextBlock(
                text="Hello",
                bbox={"x": 10, "y": 10, "width": 10, "height": 10},
                metadata={"confidence": 0.9},
            )
        ],
    )

    out = draw_ocr_bboxes(img, page, show_text=True, show_confidence=True)
    assert out.size == img.size
