from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
from PIL import Image

from extraction import extract_pdf
from comparison.table_comparison import compare_tables
from visualization.diff_renderer import overlay_diffs


def main() -> None:
    orig = Path("data/synthetic/dataset/variation_01/variation_01_original.pdf")
    mod = Path("data/synthetic/dataset/variation_01/variation_01_modified.pdf")

    print("=== Extract digital PDFs (with layout analysis) ===")
    pages_a = extract_pdf(orig)
    pages_b = extract_pdf(mod)
    print("pages:", len(pages_a), len(pages_b))

    print("=== Compare tables ===")
    diffs = compare_tables(pages_a, pages_b)
    struct = [
        d
        for d in diffs
        if (d.metadata or {}).get("type") == "table" and (d.metadata or {}).get("table_change") == "structure"
    ]

    print("table_diffs_total:", len(diffs), "structure_diffs:", len(struct))
    if not struct:
        raise SystemExit("No table structure diffs found")

    d = struct[0]
    print("page_num_a:", d.page_num, "page_num_b:", d.page_num_b)
    print("bbox_a:", d.bbox)
    print("bbox_b:", d.bbox_b)

    out_dir = Path("debug_output")
    out_dir.mkdir(exist_ok=True)

    page_a_index = d.page_num - 1
    page_b_index = (d.page_num_b or d.page_num) - 1

    doc_a = fitz.open(orig)
    doc_b = fitz.open(mod)

    pix_a = doc_a[page_a_index].get_pixmap(dpi=150)
    pix_b = doc_b[page_b_index].get_pixmap(dpi=150)

    img_a = np.frombuffer(pix_a.samples, dtype=np.uint8).reshape(pix_a.height, pix_a.width, pix_a.n)
    img_b = np.frombuffer(pix_b.samples, dtype=np.uint8).reshape(pix_b.height, pix_b.width, pix_b.n)

    if img_a.shape[2] == 4:
        img_a = img_a[:, :, :3]
    if img_b.shape[2] == 4:
        img_b = img_b[:, :, :3]

    ov_a = overlay_diffs(
        img_a,
        [d],
        page_width=float(pix_a.width),
        page_height=float(pix_a.height),
        use_normalized=True,
        use_word_bboxes=True,
        doc_side="a",
    )

    ov_b = overlay_diffs(
        img_b,
        [d],
        page_width=float(pix_b.width),
        page_height=float(pix_b.height),
        use_normalized=True,
        use_word_bboxes=True,
        doc_side="b",
    )

    out_a = out_dir / "verify_table_bbox_b_a.png"
    out_b = out_dir / "verify_table_bbox_b_b.png"

    Image.fromarray(ov_a).save(out_a)
    Image.fromarray(ov_b).save(out_b)

    print("Saved overlays:")
    print("-", out_a)
    print("-", out_b)


if __name__ == "__main__":
    main()
