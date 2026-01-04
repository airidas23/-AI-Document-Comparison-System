import io
from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageDraw

from extraction.pdf_parser import parse_pdf_words_as_lines
from comparison.figure_comparison import compare_figure_captions


def _make_test_image(*, variant: str) -> bytes:
    """Create a small image with a simple geometric difference."""
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Common border
    draw.rectangle([10, 10, 190, 190], outline=(0, 0, 0), width=3)

    # Variant-specific mark
    if variant == "A":
        draw.rectangle([40, 60, 120, 140], fill=(0, 0, 0))
        draw.text((130, 85), "A", fill=(0, 0, 0))
    elif variant == "B":
        draw.ellipse([60, 60, 140, 140], fill=(0, 0, 0))
        draw.text((40, 85), "B", fill=(0, 0, 0))
    else:
        raise ValueError(f"Unknown variant: {variant}")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_pdf_with_image(pdf_path: Path, *, image_bytes: bytes) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4-ish in points

    # Insert image in a stable rectangle
    rect = fitz.Rect(100, 200, 400, 500)
    page.insert_image(rect, stream=image_bytes)

    # Optional caption text (not required for visual comparison)
    page.insert_text((100, 520), "Figure 1: Example", fontsize=12)

    doc.save(str(pdf_path))
    doc.close()


@pytest.mark.parametrize("variant_a,variant_b,expect_visual", [("A", "B", True), ("A", "A", False)])
def test_figure_visual_hash_diff(tmp_path: Path, variant_a: str, variant_b: str, expect_visual: bool) -> None:
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"

    _write_pdf_with_image(pdf_a, image_bytes=_make_test_image(variant=variant_a))
    _write_pdf_with_image(pdf_b, image_bytes=_make_test_image(variant=variant_b))

    pages_a = parse_pdf_words_as_lines(pdf_a, run_layout_analysis=True)
    pages_b = parse_pdf_words_as_lines(pdf_b, run_layout_analysis=True)

    diffs = compare_figure_captions(pages_a, pages_b)
    visual_diffs = [d for d in diffs if d.change_type == "visual"]

    if expect_visual:
        assert visual_diffs, "Expected at least one visual diff for changed figure image"
    else:
        assert not visual_diffs, "Did not expect visual diffs when figure image is identical"
