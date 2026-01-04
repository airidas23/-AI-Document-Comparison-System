#!/usr/bin/env python3
"""Run a systematic Tesseract benchmark on a small dataset.

Outputs a reproducible run folder:
- runs/tesseract/<run_id>/meta.json
- runs/tesseract/<run_id>/scores.csv
- runs/tesseract/<run_id>/pred/*.txt
- runs/tesseract/<run_id>/artifacts/bbox/*.png (optional)
- runs/tesseract/<run_id>/artifacts/searchable/*.pdf (optional)

Designed to mirror the practical plan you outlined (baseline, image_to_data audit,
PDF rasterization, searchable PDF, preprocessing experiments, psm/oem/lang grid).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import List, Optional

from PIL import Image
from utils.logging import configure_logging, logger
from utils.tesseract_bench import (
    OcrParams,
    build_run_metadata,
    discover_samples,
    draw_bbox_debug,
    merge_searchable_pdf_pages,
    pdf_to_images,
    preprocess,
    run_tesseract_on_image,
    score_wer_cer,
    write_json,
)


def _parse_int_list(values: Optional[List[str]]) -> List[int]:
    if not values:
        return []
    out: List[int] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
    return out


def _parse_str_list(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _config_key(pre: str, dpi: Optional[int], params: OcrParams) -> str:
    bits = [pre, f"psm{params.psm}", f"oem{params.oem}", f"lang{params.lang}"]
    if dpi is not None:
        bits.insert(1, f"dpi{dpi}")
    return "__".join(bits)


def main() -> int:
    configure_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="dataset", help="Dataset root folder")
    ap.add_argument("--out-root", type=str, default="runs/tesseract", help="Where to write runs")
    ap.add_argument("--run-id", type=str, default="", help="Run id (default: timestamp+hash)")

    ap.add_argument("--preprocess", action="append", default=None, help="Variants: none, grayscale, gray_denoise, gray_binarize, gray_denoise_binarize, gray_sharpen")
    ap.add_argument("--psm", action="append", default=None, help="PSM values (comma-separated ok). Default: 6")
    ap.add_argument("--oem", action="append", default=None, help="OEM values (comma-separated ok). Default: 3")
    ap.add_argument("--lang", action="append", default=None, help="Language(s), e.g. eng, lit, lit+eng")
    ap.add_argument("--extra-config", type=str, default="", help="Extra tesseract config string (e.g. -c preserve_interword_spaces=1)")

    ap.add_argument("--dpi", action="append", default=None, help="PDF raster DPI(s). Default: 300")
    ap.add_argument("--pdf-renderer", type=str, default="auto", choices=["auto", "pdf2image", "pymupdf"], help="PDF raster backend")

    ap.add_argument("--emit-bbox", action="store_true", help="Write bbox debug images")
    ap.add_argument("--bbox-min-conf", type=float, default=0.0, help="Min conf to draw in bbox images")
    ap.add_argument("--emit-searchable-pdf", action="store_true", help="Write searchable PDFs (can be slow)")

    ap.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 = no limit)")
    ap.add_argument("--include-category", action="append", default=[], help="Only include these top-level categories")

    args = ap.parse_args()

    dataset_root = Path(args.dataset)
    out_root = Path(args.out_root)

    preprocess_list = _parse_str_list(args.preprocess) or [
        "none",
        "grayscale",
        "gray_denoise",
        "gray_binarize",
        "gray_denoise_binarize",
        "gray_sharpen",
    ]
    psm_list = _parse_int_list(args.psm) or [6]
    oem_list = _parse_int_list(args.oem) or [3]
    lang_list = _parse_str_list(args.lang) or ["eng"]
    dpi_list = _parse_int_list(args.dpi) or [300]

    include_cats = set(_parse_str_list(args.include_category))

    samples = discover_samples(dataset_root)
    if include_cats:
        samples = [s for s in samples if s.category in include_cats]

    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    if not samples:
        logger.error("No samples found under %s", dataset_root)
        return 2

    run_id = (args.run_id or "").strip()
    if not run_id:
        run_id = f"{time_tag()}_{_sha1(str(dataset_root.resolve()))}"

    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = build_run_metadata(vars(args))
    write_json(run_dir / "meta.json", meta)

    scores_path = run_dir / "scores.csv"
    pred_dir = run_dir / "pred"
    bbox_dir = run_dir / "artifacts" / "bbox"
    searchable_dir = run_dir / "artifacts" / "searchable"

    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "category",
                "input_path",
                "input_type",
                "page_index",
                "preprocess",
                "dpi",
                "lang",
                "psm",
                "oem",
                "extra_config",
                "wer",
                "cer",
                "avg_conf",
                "median_conf",
                "pct_conf_ge_80",
                "n_words",
                "elapsed_sec",
                "tesseract_version",
                "ok",
                "error",
            ],
        )
        writer.writeheader()

        total_jobs = len(samples) * len(preprocess_list) * len(lang_list) * len(psm_list) * len(oem_list)
        logger.info("Found %d samples (%d jobs before PDF page expansion)", len(samples), total_jobs)

        for sample in samples:
            input_path = sample.input_path

            gt_text = ""
            if sample.gt_path and sample.gt_path.exists():
                gt_text = sample.gt_path.read_text(encoding="utf-8", errors="replace")

            if input_path.suffix.lower() == ".pdf":
                for dpi in dpi_list:
                    pages = pdf_to_images(input_path, dpi=dpi, renderer=args.pdf_renderer)
                    run_pdf_sample(
                        writer,
                        sample,
                        pages,
                        gt_text,
                        preprocess_list,
                        lang_list,
                        psm_list,
                        oem_list,
                        args.extra_config,
                        dpi,
                        pred_dir,
                        bbox_dir if args.emit_bbox else None,
                        args.bbox_min_conf,
                        searchable_dir if args.emit_searchable_pdf else None,
                    )
            else:
                img = Image.open(input_path).convert("RGB")
                run_image_sample(
                    writer,
                    sample,
                    img,
                    gt_text,
                    preprocess_list,
                    lang_list,
                    psm_list,
                    oem_list,
                    args.extra_config,
                    None,
                    pred_dir,
                    bbox_dir if args.emit_bbox else None,
                    args.bbox_min_conf,
                    searchable_dir if args.emit_searchable_pdf else None,
                )

    logger.info("Done. Results in %s", run_dir)
    logger.info("- %s", scores_path)
    return 0


def time_tag() -> str:
    import time

    return time.strftime("%Y%m%d_%H%M%S")


def run_image_sample(
    writer: csv.DictWriter,
    sample,
    img: Image.Image,
    gt_text: str,
    preprocess_list: List[str],
    lang_list: List[str],
    psm_list: List[int],
    oem_list: List[int],
    extra_config: str,
    dpi: Optional[int],
    pred_dir: Path,
    bbox_dir: Optional[Path],
    bbox_min_conf: float,
    searchable_dir: Optional[Path],
) -> None:
    for pre in preprocess_list:
        try:
            img_p = preprocess(img, pre)
        except Exception as exc:
            logger.warning("Preprocess '%s' failed for %s: %s", pre, sample.sample_id, exc)
            continue

        for lang in lang_list:
            for psm in psm_list:
                for oem in oem_list:
                    params = OcrParams(lang=lang, psm=psm, oem=oem, extra_config=extra_config)
                    key = _config_key(pre, dpi, params)

                    res, data, pdf_bytes = run_tesseract_on_image(
                        img_p,
                        params,
                        emit_searchable_pdf=bool(searchable_dir),
                    )

                    # Write pred
                    pred_path = pred_dir / sample.sample_id
                    pred_path.mkdir(parents=True, exist_ok=True)
                    (pred_path / f"{key}.txt").write_text(res.text, encoding="utf-8", errors="replace")

                    # BBox debug
                    if bbox_dir is not None and data is not None:
                        out_img = bbox_dir / sample.sample_id
                        out_img.mkdir(parents=True, exist_ok=True)
                        draw_bbox_debug(
                            img_p,
                            data,
                            out_img / f"{key}.png",
                            min_conf=bbox_min_conf,
                        )

                    # Searchable PDF
                    if searchable_dir is not None and pdf_bytes is not None:
                        out_pdf = searchable_dir / sample.sample_id
                        out_pdf.mkdir(parents=True, exist_ok=True)
                        (out_pdf / f"{key}.pdf").write_bytes(pdf_bytes)

                    score = score_wer_cer(gt_text, res.text) if gt_text else None

                    writer.writerow(
                        {
                            "sample_id": sample.sample_id,
                            "category": sample.category,
                            "input_path": str(sample.input_path),
                            "input_type": sample.input_path.suffix.lower().lstrip("."),
                            "page_index": 0,
                            "preprocess": pre,
                            "dpi": dpi or "",
                            "lang": lang,
                            "psm": psm,
                            "oem": oem,
                            "extra_config": extra_config,
                            "wer": score.wer if score else "",
                            "cer": score.cer if score else "",
                            "avg_conf": res.conf_stats.avg_conf,
                            "median_conf": res.conf_stats.median_conf,
                            "pct_conf_ge_80": res.conf_stats.pct_conf_ge_80,
                            "n_words": res.conf_stats.n_words,
                            "elapsed_sec": res.elapsed_sec,
                            "tesseract_version": res.tesseract_version,
                            "ok": res.ok,
                            "error": res.error,
                        }
                    )


def run_pdf_sample(
    writer: csv.DictWriter,
    sample,
    pages: List[Image.Image],
    gt_text: str,
    preprocess_list: List[str],
    lang_list: List[str],
    psm_list: List[int],
    oem_list: List[int],
    extra_config: str,
    dpi: int,
    pred_dir: Path,
    bbox_dir: Optional[Path],
    bbox_min_conf: float,
    searchable_dir: Optional[Path],
) -> None:
    # For PDFs, WER/CER is only meaningful if GT exists and is page-aligned.
    # We still compute global (whole PDF) on concatenated output if gt_text exists.

    for pre in preprocess_list:
        for lang in lang_list:
            for psm in psm_list:
                for oem in oem_list:
                    params = OcrParams(lang=lang, psm=psm, oem=oem, extra_config=extra_config)
                    key = _config_key(pre, dpi, params)

                    page_texts: List[str] = []
                    page_pdfs: List[bytes] = []

                    for pi, img in enumerate(pages):
                        try:
                            img_p = preprocess(img, pre)
                        except Exception as exc:
                            logger.warning("Preprocess '%s' failed for %s p%d: %s", pre, sample.sample_id, pi, exc)
                            continue

                        res, data, pdf_bytes = run_tesseract_on_image(
                            img_p,
                            params,
                            emit_searchable_pdf=bool(searchable_dir),
                        )

                        page_texts.append(res.text)

                        # Write per-page pred
                        pred_path = pred_dir / sample.sample_id
                        pred_path.mkdir(parents=True, exist_ok=True)
                        (pred_path / f"{key}__p{pi+1:03d}.txt").write_text(res.text, encoding="utf-8", errors="replace")

                        if bbox_dir is not None and data is not None:
                            out_img = bbox_dir / sample.sample_id
                            out_img.mkdir(parents=True, exist_ok=True)
                            draw_bbox_debug(
                                img_p,
                                data,
                                out_img / f"{key}__p{pi+1:03d}.png",
                                min_conf=bbox_min_conf,
                            )

                        if searchable_dir is not None and pdf_bytes is not None:
                            page_pdfs.append(pdf_bytes)

                        # Per-page row
                        writer.writerow(
                            {
                                "sample_id": sample.sample_id,
                                "category": sample.category,
                                "input_path": str(sample.input_path),
                                "input_type": "pdf",
                                "page_index": pi + 1,
                                "preprocess": pre,
                                "dpi": dpi,
                                "lang": lang,
                                "psm": psm,
                                "oem": oem,
                                "extra_config": extra_config,
                                "wer": "",
                                "cer": "",
                                "avg_conf": res.conf_stats.avg_conf,
                                "median_conf": res.conf_stats.median_conf,
                                "pct_conf_ge_80": res.conf_stats.pct_conf_ge_80,
                                "n_words": res.conf_stats.n_words,
                                "elapsed_sec": res.elapsed_sec,
                                "tesseract_version": res.tesseract_version,
                                "ok": res.ok,
                                "error": res.error,
                            }
                        )

                    # Optional merged searchable PDF for the whole document
                    if searchable_dir is not None and page_pdfs:
                        merged_bytes = merge_searchable_pdf_pages(page_pdfs)
                        out_pdf = searchable_dir / sample.sample_id
                        out_pdf.mkdir(parents=True, exist_ok=True)
                        (out_pdf / f"{key}__merged.pdf").write_bytes(merged_bytes)

                    # Optional overall WER/CER on concatenated text (if GT exists but not page-aligned)
                    if gt_text:
                        pred_all = "\n\n".join(page_texts)
                        score = score_wer_cer(gt_text, pred_all)
                        writer.writerow(
                            {
                                "sample_id": sample.sample_id,
                                "category": sample.category,
                                "input_path": str(sample.input_path),
                                "input_type": "pdf",
                                "page_index": "ALL",
                                "preprocess": pre,
                                "dpi": dpi,
                                "lang": lang,
                                "psm": psm,
                                "oem": oem,
                                "extra_config": extra_config,
                                "wer": score.wer,
                                "cer": score.cer,
                                "avg_conf": "",
                                "median_conf": "",
                                "pct_conf_ge_80": "",
                                "n_words": "",
                                "elapsed_sec": "",
                                "tesseract_version": "",
                                "ok": True,
                                "error": "",
                            }
                        )


if __name__ == "__main__":
    raise SystemExit(main())
