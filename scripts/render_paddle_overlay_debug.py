"""Debug helper: run Paddle OCR comparison and render page-1 overlays.

Usage:
  ./.venv/bin/python scripts/render_paddle_overlay_debug.py

Writes:
  debug_output/ui_overlay_paddle_all_A_page_1.png
  debug_output/ui_overlay_paddle_all_B_page_1.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.compare_pdfs import compare_pdfs  # noqa: E402
from visualization.pdf_viewer import render_pages  # noqa: E402


def main() -> int:
    pdf_a = Path(
        "data/synthetic/test_scanned_dataset/variation_01/variation_01_original_scanned.pdf"
    )
    pdf_b = Path(
        "data/synthetic/test_scanned_dataset/variation_01/variation_01_modified_scanned.pdf"
    )

    res = compare_pdfs(
        str(pdf_a),
        str(pdf_b),
        ocr_mode="ocr_only",
        ocr_engine="paddle",
        force_ocr=True,
        debug_mode=False,
    )
    diffs = res.diffs or []

    min_w = 1.0
    min_h = 1.0
    cnt = 0
    for d in diffs:
        md = d.metadata or {}
        for key in ("word_bboxes_a", "word_bboxes_b", "word_bboxes"):
            for b in md.get(key) or []:
                if not isinstance(b, dict):
                    continue
                w = float(b.get("width", 0.0))
                h = float(b.get("height", 0.0))
                if w > 0.0 and h > 0.0:
                    cnt += 1
                    min_w = min(min_w, w)
                    min_h = min(min_h, h)

    print("diffs_total", len(diffs))
    print("word_bbox_count", cnt)
    print("min_norm_w", round(min_w, 6))
    print("min_norm_h", round(min_h, 6))

    out_dir = Path("debug_output")
    out_dir.mkdir(exist_ok=True)

    imgs_a = render_pages(pdf_a, dpi=144, diffs=diffs, scale_factor=2.0, doc_side="a")
    imgs_b = render_pages(pdf_b, dpi=144, diffs=diffs, scale_factor=2.0, doc_side="b")

    out_a = out_dir / "ui_overlay_paddle_all_A_page_1.png"
    out_b = out_dir / "ui_overlay_paddle_all_B_page_1.png"

    for p, img in imgs_a:
        if p == 1:
            Image.fromarray(img).save(out_a)
            break
    for p, img in imgs_b:
        if p == 1:
            Image.fromarray(img).save(out_b)
            break

    print("saved_A", out_a)
    print("saved_B", out_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
