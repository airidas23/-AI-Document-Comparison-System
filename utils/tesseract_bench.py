"""Tesseract (pytesseract) benchmark helpers.

Goal: provide a small, reproducible harness to evaluate OCR quality for *your* documents.

Features:
- Preprocessing variants (DataCamp-style A–F)
- image_to_string baseline + image_to_data (bbox + confidence)
- Optional searchable PDF generation via image_to_pdf_or_hocr
- PDF -> images via pdf2image (Poppler) or PyMuPDF fallback
- WER/CER scoring (jiwer)

This module is intentionally independent from the main pipeline.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from utils.logging import logger


@dataclass(frozen=True)
class SampleRef:
    sample_id: str
    category: str
    input_path: Path
    gt_path: Optional[Path]


@dataclass(frozen=True)
class OcrParams:
    lang: str
    psm: int
    oem: int
    extra_config: str = ""

    def to_tesseract_config(self) -> str:
        extra = (self.extra_config or "").strip()
        # Avoid accidental duplication.
        for token in ("--psm", "--oem"):
            if token in extra:
                logger.warning("extra_config contains %s; it will be kept as-is", token)
        parts = [f"--psm {self.psm}", f"--oem {self.oem}"]
        if extra:
            parts.append(extra)
        return " ".join(parts).strip()


@dataclass
class ConfStats:
    n_words: int = 0
    n_empty: int = 0
    avg_conf: float = 0.0
    median_conf: float = 0.0
    pct_conf_ge_80: float = 0.0


@dataclass
class OcrRunResult:
    ok: bool
    text: str = ""
    elapsed_sec: float = 0.0
    conf_stats: ConfStats = field(default_factory=ConfStats)
    tesseract_version: str = ""
    error: str = ""


@dataclass
class ScoreResult:
    wer: Optional[float] = None
    cer: Optional[float] = None


# -----------------------------------------------------------------------------
# Dataset discovery
# -----------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_PDF_EXTS = {".pdf"}


def _rel_category(dataset_root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(dataset_root)
    except ValueError:
        return ""
    parts = rel.parts
    return parts[0] if parts else ""


def discover_samples(dataset_root: Path) -> List[SampleRef]:
    """Discover samples in a flexible dataset layout.

    Supports either:
    - folder sample: <any>/gt.txt + (input.(png|pdf|...)) in same folder
    - file pair: <any>/<name>.(png|pdf) + <any>/<name>.gt.txt or <name>.txt
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    samples: List[SampleRef] = []

    # Folder-based: look for gt.txt
    for gt in dataset_root.rglob("gt.txt"):
        folder = gt.parent
        candidates = [p for p in folder.iterdir() if p.suffix.lower() in (_IMAGE_EXTS | _PDF_EXTS)]
        if not candidates:
            continue
        input_path = sorted(candidates)[0]
        rel = folder.relative_to(dataset_root)
        category = rel.parts[0] if rel.parts else ""
        sample_id = str(rel).replace(os.sep, "__")
        samples.append(SampleRef(sample_id=sample_id, category=category, input_path=input_path, gt_path=gt))

    # File-pair-based
    for inp in dataset_root.rglob("*"):
        if not inp.is_file():
            continue
        ext = inp.suffix.lower()
        if ext not in (_IMAGE_EXTS | _PDF_EXTS):
            continue
        # Skip if already captured via gt.txt folder rule
        if any(s.input_path == inp for s in samples):
            continue

        gt1 = inp.with_suffix(".gt.txt")
        gt2 = inp.with_suffix(".txt")
        gt_path = gt1 if gt1.exists() else (gt2 if gt2.exists() else None)

        rel = inp.relative_to(dataset_root)
        category = rel.parts[0] if rel.parts else ""
        sample_id = str(rel.with_suffix("")).replace(os.sep, "__")
        samples.append(SampleRef(sample_id=sample_id, category=category, input_path=inp, gt_path=gt_path))

    samples = sorted(samples, key=lambda s: s.sample_id)
    return samples


# -----------------------------------------------------------------------------
# Preprocessing variants (DataCamp-style)
# -----------------------------------------------------------------------------

def _to_cv2_gray(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    # RGB -> gray
    return (0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]).astype(np.uint8)


def preprocess(img: Image.Image, variant: str) -> Image.Image:
    """Apply a named preprocessing variant (A–F style).

    Variants:
    - none
    - grayscale
    - gray_denoise
    - gray_binarize
    - gray_denoise_binarize
    - gray_sharpen
    """
    variant = (variant or "none").strip().lower()

    if variant == "none":
        return img

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for preprocessing") from exc

    if variant == "grayscale":
        gray = _to_cv2_gray(img)
        return Image.fromarray(gray)

    if variant == "gray_denoise":
        gray = _to_cv2_gray(img)
        den = cv2.medianBlur(gray, 3)
        return Image.fromarray(den)

    if variant == "gray_binarize":
        gray = _to_cv2_gray(img)
        # Adaptive threshold tends to be robust for uneven lighting.
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return Image.fromarray(th)

    if variant == "gray_denoise_binarize":
        gray = _to_cv2_gray(img)
        den = cv2.medianBlur(gray, 3)
        th = cv2.adaptiveThreshold(
            den,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return Image.fromarray(th)

    if variant == "gray_sharpen":
        gray = _to_cv2_gray(img)
        # Unsharp mask
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
        return Image.fromarray(sharp)

    raise ValueError(f"Unknown preprocessing variant: {variant}")


# -----------------------------------------------------------------------------
# Tesseract run + metrics
# -----------------------------------------------------------------------------

def get_tesseract_version() -> str:
    try:
        import pytesseract

        return str(pytesseract.get_tesseract_version())
    except Exception as exc:
        return f"unknown ({exc})"


def compute_conf_stats(data: Dict[str, Any]) -> ConfStats:
    """Compute word-level confidence stats from pytesseract image_to_data dict."""
    if not data or "text" not in data:
        return ConfStats()

    texts = data.get("text", [])
    confs = data.get("conf", [])
    levels = data.get("level", [])

    word_confs: List[float] = []
    n_empty = 0

    for i in range(min(len(texts), len(confs))):
        text = str(texts[i] or "").strip()
        if not text:
            n_empty += 1
            continue

        # Tesseract convention: level=5 is word
        if i < len(levels):
            try:
                if int(levels[i]) != 5:
                    continue
            except Exception:
                pass

        try:
            c = float(confs[i])
        except Exception:
            continue

        if c < 0:
            # -1 often indicates non-word items
            continue
        word_confs.append(max(0.0, min(100.0, c)))

    if not word_confs:
        return ConfStats(n_words=0, n_empty=n_empty)

    avg = float(sum(word_confs) / len(word_confs))
    med = float(statistics.median(word_confs))
    pct80 = float(sum(1 for c in word_confs if c >= 80.0) / len(word_confs))
    return ConfStats(
        n_words=len(word_confs),
        n_empty=n_empty,
        avg_conf=avg,
        median_conf=med,
        pct_conf_ge_80=pct80,
    )


def run_tesseract_on_image(
    img: Image.Image,
    params: OcrParams,
    *,
    emit_searchable_pdf: bool = False,
) -> Tuple[OcrRunResult, Optional[Dict[str, Any]], Optional[bytes]]:
    """Run Tesseract baseline OCR + data.

    Returns:
        (run_result, image_to_data_dict, pdf_bytes)
    """
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        raise RuntimeError("pytesseract is required (and tesseract binary installed)") from exc

    cfg = params.to_tesseract_config()
    t0 = time.perf_counter()
    try:
        text = pytesseract.image_to_string(img, lang=params.lang, config=cfg)
        data = pytesseract.image_to_data(img, lang=params.lang, config=cfg, output_type=Output.DICT)
        pdf_bytes = None
        if emit_searchable_pdf:
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, lang=params.lang, config=cfg, extension="pdf")
        elapsed = time.perf_counter() - t0
        stats = compute_conf_stats(data)
        return (
            OcrRunResult(
                ok=True,
                text=text or "",
                elapsed_sec=elapsed,
                conf_stats=stats,
                tesseract_version=get_tesseract_version(),
            ),
            data,
            pdf_bytes,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return (
            OcrRunResult(
                ok=False,
                text="",
                elapsed_sec=elapsed,
                conf_stats=ConfStats(),
                tesseract_version=get_tesseract_version(),
                error=str(exc),
            ),
            None,
            None,
        )


def score_wer_cer(gt_text: str, pred_text: str) -> ScoreResult:
    gt_text = gt_text or ""
    pred_text = pred_text or ""

    try:
        from jiwer import wer, cer

        return ScoreResult(wer=float(wer(gt_text, pred_text)), cer=float(cer(gt_text, pred_text)))
    except Exception:
        # Fallback: approximate CER from edit distance.
        try:
            from rapidfuzz.distance.Levenshtein import distance as lev_distance

            d = lev_distance(gt_text, pred_text)
            denom = max(len(gt_text), 1)
            return ScoreResult(wer=None, cer=float(d / denom))
        except Exception:
            return ScoreResult(wer=None, cer=None)


def draw_bbox_debug(
    img: Image.Image,
    data: Dict[str, Any],
    out_path: Path,
    *,
    min_conf: float = 0.0,
    max_boxes: int = 400,
) -> None:
    """Render word boxes from image_to_data output."""
    if not data or "text" not in data:
        return

    texts = data.get("text", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])
    confs = data.get("conf", [])
    levels = data.get("level", [])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    drawn = 0
    for i in range(len(texts)):
        if drawn >= max_boxes:
            break

        text = str(texts[i] or "").strip()
        if not text:
            continue

        # Only word level
        if i < len(levels):
            try:
                if int(levels[i]) != 5:
                    continue
            except Exception:
                pass

        try:
            conf = float(confs[i]) if i < len(confs) else -1.0
        except Exception:
            conf = -1.0
        if conf < 0 or conf < min_conf:
            continue

        try:
            x = int(lefts[i])
            y = int(tops[i])
            w = int(widths[i])
            h = int(heights[i])
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue

        # Color: red (low) -> green (high)
        g = int(max(0, min(255, (conf / 100.0) * 255)))
        r = 255 - g
        draw.rectangle([(x, y), (x + w, y + h)], outline=(r, g, 0), width=2)
        drawn += 1

    canvas.save(out_path)


# -----------------------------------------------------------------------------
# PDF -> Images + searchable PDF merge
# -----------------------------------------------------------------------------

def pdf_to_images(
    pdf_path: Path,
    *,
    dpi: int,
    renderer: str = "auto",
) -> List[Image.Image]:
    """Rasterize PDF to PIL images.

    renderer:
    - "pdf2image": uses Poppler via pdf2image
    - "pymupdf": uses fitz rendering
    - "auto": try pdf2image, fallback to pymupdf
    """
    renderer = (renderer or "auto").strip().lower()
    pdf_path = Path(pdf_path)

    if renderer in ("auto", "pdf2image"):
        try:
            from pdf2image import convert_from_path  # type: ignore

            images = convert_from_path(str(pdf_path), dpi=dpi, fmt="png")
            return [img.convert("RGB") for img in images]
        except Exception as exc:
            if renderer == "pdf2image":
                raise
            logger.warning("pdf2image failed (%s); falling back to PyMuPDF", exc)

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        out: List[Image.Image] = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            out.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        doc.close()
        return out
    except Exception as exc:
        raise RuntimeError(f"Failed to render PDF via {renderer}: {exc}") from exc


def merge_searchable_pdf_pages(page_pdfs: Sequence[bytes]) -> bytes:
    """Merge per-page searchable PDF bytes into a single PDF (via PyMuPDF)."""
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError("PyMuPDF is required to merge searchable PDF") from exc

    out_doc = fitz.open()
    for b in page_pdfs:
        src = fitz.open("pdf", b)
        out_doc.insert_pdf(src)
        src.close()
    merged = out_doc.tobytes()
    out_doc.close()
    return merged


# -----------------------------------------------------------------------------
# Run metadata
# -----------------------------------------------------------------------------

def build_run_metadata(args: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "tesseract_version": get_tesseract_version(),
        "args": args,
    }

    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        meta["pip_freeze"] = freeze.splitlines()
    except Exception:
        meta["pip_freeze"] = []

    return meta


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
