"""Tesseract OCR engine for OCR processing with bbox support."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageFilter

from comparison.models import PageData, TextBlock
from config.settings import settings
from utils.logging import logger

DEFAULT_PSM_MODE = 3
DEFAULT_OEM_MODE = 3
DEFAULT_MIN_CONFIDENCE = 30


def _pixmap_to_pil(pix) -> Image.Image:
    """Convert a PyMuPDF pixmap to a correct PIL image.

    PyMuPDF pixmaps can be grayscale (n==1), RGB (n==3), or RGBA (alpha==True).
    Treating everything as raw RGB can corrupt the byte stream and produce
    "almost blank" / "garbled" images, leading to near-empty OCR (e.g., 8 chars).
    """
    try:
        alpha = bool(getattr(pix, "alpha", 0))
        n = int(getattr(pix, "n", 0) or 0)
    except Exception:
        alpha = False
        n = 0

    if alpha:
        img = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
        return img.convert("RGB")
    if n == 1:
        return Image.frombytes("L", (pix.width, pix.height), pix.samples)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


@lru_cache()
def _get_tesseract_version() -> str:
    """Return cached Tesseract version string."""
    import pytesseract

    return str(pytesseract.get_tesseract_version())


def _coerce_int(value: object, default: int) -> int:
    """Best-effort int conversion with default fallback."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _sanitize_psm_mode(value: Optional[int]) -> int:
    """Validate PSM value (0-13)."""
    mode = _coerce_int(value, DEFAULT_PSM_MODE)
    if 0 <= mode <= 13:
        return mode
    logger.warning("Invalid Tesseract PSM mode %s. Falling back to %d.", value, DEFAULT_PSM_MODE)
    return DEFAULT_PSM_MODE


def _sanitize_oem_mode(value: Optional[int]) -> int:
    """Validate OEM value (0-3)."""
    mode = _coerce_int(value, DEFAULT_OEM_MODE)
    if 0 <= mode <= 3:
        return mode
    logger.warning("Invalid Tesseract OEM mode %s. Falling back to %d.", value, DEFAULT_OEM_MODE)
    return DEFAULT_OEM_MODE


def _sanitize_min_confidence(value: Optional[int]) -> int:
    """Clamp minimum confidence to [0, 100]."""
    conf = _coerce_int(value, DEFAULT_MIN_CONFIDENCE)
    return max(0, min(100, conf))


def _sanitize_granularity(value: Optional[str]) -> str:
    """Validate granularity setting."""
    allowed = {"word", "line", "paragraph", "block"}
    if value and value in allowed:
        return value
    if value:
        logger.warning("Invalid Tesseract granularity %s. Falling back to 'word'.", value)
    return "word"


def _build_tesseract_config(psm_mode: int, oem_mode: int, extra_config: str, *, dpi: int = 300) -> str:
    """Build Tesseract config string including PSM/OEM/DPI and optional overrides.
    
    The extra_config should include settings like:
    - '-c preserve_interword_spaces=1' for better column/table handling
    - Other tesseract configuration variables
    
    Args:
        psm_mode: Page segmentation mode (0-13)
        oem_mode: OCR engine mode (0-3)
        extra_config: Additional config string
        dpi: DPI hint for Tesseract layout analysis
    """
    parts = [f"--psm {psm_mode}", f"--oem {oem_mode}", f"--dpi {dpi}"]
    if extra_config:
        # Ensure extra_config doesn't duplicate PSM/OEM/DPI
        extra_cleaned = extra_config.strip()
        # Remove any accidental psm/oem/dpi in extra_config (user should use settings)
        import re
        extra_cleaned = re.sub(r'--psm\s+\d+', '', extra_cleaned)
        extra_cleaned = re.sub(r'--oem\s+\d+', '', extra_cleaned)
        extra_cleaned = re.sub(r'--dpi\s+\d+', '', extra_cleaned)
        extra_cleaned = extra_cleaned.strip()
        if extra_cleaned:
            parts.append(extra_cleaned)
    return " ".join(part for part in parts if part).strip()


def _parse_confidence(conf_raw: object) -> Tuple[int, float]:
    """Parse Tesseract confidence, returning (0-100 int, 0-1 float)."""
    try:
        conf_i = int(float(conf_raw))
    except (TypeError, ValueError):
        conf_i = 0
    if conf_i < 0:
        conf_i = 0
    conf = max(0.0, min(1.0, conf_i / 100.0))
    return conf_i, conf


def _build_bbox(
    left: object,
    top: object,
    width: object,
    height: object,
    scale_factor: float,
) -> Optional[Dict[str, float]]:
    """Convert pixel bbox to point bbox, returning None if invalid."""
    try:
        w = float(width)
        h = float(height)
        x = float(left)
        y = float(top)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return {
        "x": x * scale_factor,
        "y": y * scale_factor,
        "width": w * scale_factor,
        "height": h * scale_factor,
    }


def _merge_bboxes(bboxes: Iterable[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Merge multiple bboxes into a single enclosing bbox."""
    bboxes = list(bboxes)
    if not bboxes:
        return None
    x0 = min(b["x"] for b in bboxes)
    y0 = min(b["y"] for b in bboxes)
    x1 = max(b["x"] + b["width"] for b in bboxes)
    y1 = max(b["y"] + b["height"] for b in bboxes)
    if x1 <= x0 or y1 <= y0:
        return None
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def _preprocess_for_tesseract(img: Image.Image) -> Image.Image:
    """Preprocess image for better Tesseract OCR accuracy.
    
    Applies (default = mild, safe):
    - Grayscale conversion
    - Auto-contrast enhancement
    - Light median denoise

    Optional (disabled by default):
    - Otsu binarization

    Notes:
        A previous histogram-quantile binarization approach could push the
        threshold into the extreme highlights (e.g. ~245), turning most pixels
        black and causing Tesseract to return near-empty results.
    """
    # Allow disabling preprocessing for debugging.
    if not bool(getattr(settings, "tesseract_preprocess_enabled", True)):
        return img.convert("L")

    # Convert to grayscale
    g = img.convert("L")
    # Auto-contrast to normalize brightness
    g = ImageOps.autocontrast(g)
    # Light denoise with median filter
    median_size = int(getattr(settings, "tesseract_preprocess_median_size", 3) or 3)
    if median_size < 1:
        median_size = 1
    if median_size % 2 == 0:
        median_size += 1
    g = g.filter(ImageFilter.MedianFilter(size=median_size))

    # Mild edge sharpening can help scans with slightly blurred glyph edges.
    if bool(getattr(settings, "tesseract_preprocess_unsharp", True)):
        try:
            g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        except Exception:
            # UnsharpMask may be unavailable in some minimal Pillow builds.
            pass
    enable_binarize = bool(getattr(settings, "tesseract_preprocess_binarize", False))
    if not enable_binarize:
        if bool(getattr(settings, "tesseract_invert", False)):
            return ImageOps.invert(g)
        return g

    # Otsu thresholding (robust, deterministic) for challenging scans.
    arr = np.asarray(g, dtype=np.uint8)
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = arr.size
    if total <= 0:
        return g

    sum_total = np.dot(np.arange(256, dtype=np.float64), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += float(t) * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    # Keep black text on white background.
    out = g.point(lambda p: 255 if p > threshold else 0)
    if bool(getattr(settings, "tesseract_invert", False)):
        out = ImageOps.invert(out)
    return out


def _tighten_bbox_to_ink(
    img_l: Image.Image,
    bbox_pt: Dict[str, float],
    *,
    dpi: int,
    pad_px: int = 1,
) -> Dict[str, float]:
    """Tighten bbox to actual ink (dark pixels) within the region.
    
    Tesseract word bboxes often include whitespace around text.
    This function crops to the actual ink extent for more precise highlighting.
    
    Args:
        img_l: Grayscale image (mode "L")
        bbox_pt: Bbox in PDF points {x, y, width, height}
        dpi: Render DPI for coordinate conversion
        pad_px: Padding pixels around detected ink
    
    Returns:
        Tightened bbox in PDF points
    """
    # Convert pt -> px
    scale = dpi / 72.0
    x0 = int(bbox_pt["x"] * scale)
    y0 = int(bbox_pt["y"] * scale)
    x1 = int((bbox_pt["x"] + bbox_pt["width"]) * scale)
    y1 = int((bbox_pt["y"] + bbox_pt["height"]) * scale)
    
    # Clamp to image bounds
    x0 = max(0, min(x0, img_l.width - 1))
    y0 = max(0, min(y0, img_l.height - 1))
    x1 = max(1, min(x1, img_l.width))
    y1 = max(1, min(y1, img_l.height))
    
    if x1 <= x0 or y1 <= y0:
        return bbox_pt
    
    # Crop region and find ink pixels
    crop = img_l.crop((x0, y0, x1, y1))
    arr = np.array(crop)
    
    # "Ink" = dark pixels (< 200 in grayscale)
    ink_mask = arr < 200
    ink_coords = np.where(ink_mask)
    
    if ink_coords[0].size == 0 or ink_coords[1].size == 0:
        return bbox_pt  # No ink found, keep original
    
    ys, xs = ink_coords
    # Calculate tight bounds with padding
    nx0 = max(0, x0 + int(xs.min()) - pad_px)
    ny0 = max(0, y0 + int(ys.min()) - pad_px)
    nx1 = min(img_l.width, x0 + int(xs.max()) + 1 + pad_px)
    ny1 = min(img_l.height, y0 + int(ys.max()) + 1 + pad_px)
    
    # Convert px -> pt
    inv_scale = 72.0 / dpi
    return {
        "x": nx0 * inv_scale,
        "y": ny0 * inv_scale,
        "width": (nx1 - nx0) * inv_scale,
        "height": (ny1 - ny0) * inv_scale,
    }


def _group_word_entries(
    word_entries: List[dict],
    granularity: str,
) -> List[dict]:
    """Group word entries into lines/paragraphs/blocks."""
    if granularity == "line":
        key_fields = ("block_num", "par_num", "line_num")
    elif granularity == "paragraph":
        key_fields = ("block_num", "par_num")
    else:
        key_fields = ("block_num",)

    grouped: dict[Tuple[int, ...], List[dict]] = {}
    for entry in word_entries:
        key = tuple(entry.get(field, 0) for field in key_fields)
        grouped.setdefault(key, []).append(entry)

    out = []
    for key, words in grouped.items():
        words_sorted = sorted(words, key=lambda w: (w.get("word_num", 0), w.get("order", 0)))
        text = " ".join(w["text"] for w in words_sorted if w.get("text"))
        bboxes = [w["bbox"] for w in words_sorted if w.get("bbox")]
        merged_bbox = _merge_bboxes(bboxes)
        if not merged_bbox or not text:
            continue
        avg_conf = sum(w["conf"] for w in words_sorted) / max(len(words_sorted), 1)
        out.append({
            "text": text,
            "bbox": merged_bbox,
            "confidence": avg_conf,
            "words": [
                {"text": w["text"], "bbox": w["bbox"], "conf": w["conf"]}
                for w in words_sorted
            ],
            "group_key": key,
        })
    return out


def _tesseract_boxes_to_text_blocks(
    boxes_text: str,
    ocr_text: str,
    *,
    dpi: int,
    image_height: int,
) -> List[TextBlock]:
    """
    Convert Tesseract image_to_boxes output to TextBlocks.

    This is a fallback path that approximates word boxes by grouping character boxes
    based on whitespace in the OCR text. Confidence is not available in this format.
    """
    if dpi <= 0:
        dpi = 150
    scale_factor = 72.0 / float(dpi)
    if not boxes_text:
        return []

    boxes = []
    for line in boxes_text.splitlines():
        parts = line.split(" ")
        if len(parts) < 5:
            continue
        char = parts[0]
        left = _coerce_int(parts[1], 0)
        bottom = _coerce_int(parts[2], 0)
        right = _coerce_int(parts[3], 0)
        top = _coerce_int(parts[4], 0)
        # Convert bottom-left origin to top-left origin.
        y_top = image_height - top
        y_bottom = image_height - bottom
        bbox = _build_bbox(left, y_top, right - left, y_bottom - y_top, scale_factor)
        if bbox is None:
            continue
        boxes.append({"char": char, "bbox": bbox})

    if not boxes:
        return []

    text_blocks = []
    words = [w for w in ocr_text.split() if w]
    box_idx = 0
    for word in words:
        if box_idx >= len(boxes):
            break
        char_boxes = boxes[box_idx: box_idx + len(word)]
        box_idx += len(word)
        merged_bbox = _merge_bboxes([cb["bbox"] for cb in char_boxes])
        if not merged_bbox:
            continue
        block = TextBlock(
            text=word,
            bbox=merged_bbox,
            style=None,
            metadata={
                "granularity": "word",
                "level": "word",
                "ocr_engine": "tesseract",
                "bbox_source": "approx",
                "bbox_units": "pt",
                "bbox_space": "page",
                "confidence": 0.0,
                "confidence_unavailable": True,
                "text_source": "tesseract_ocr",
            },
        )
        text_blocks.append(block)

    return text_blocks


def ocr_pdf(path: str | Path) -> List[PageData]:
    """
    Process a PDF through Tesseract OCR and return PageData.

    Args:
        path: Path to PDF file

    Returns:
        List of PageData objects with extracted text blocks
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    total_start = time.time()
    path = Path(path)
    logger.info("[Tesseract] Running OCR on PDF: %s", path)

    try:
        import pytesseract
        from pytesseract import Output, TesseractError
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is required. Install via `pip install pytesseract`. "
            "Also ensure Tesseract binary is installed (brew install tesseract on Mac)."
        ) from exc

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for OCR rendering. Install via `pip install PyMuPDF`."
        ) from exc

    try:
        version = _get_tesseract_version()
        logger.debug("[Tesseract] Version: %s", version)
    except Exception as exc:
        raise RuntimeError(
            "Tesseract binary not found. Install Tesseract (brew install tesseract on Mac)."
        ) from exc

    doc = fitz.open(path)
    total_pages = len(doc)

    # Use configurable DPI (default 300 for accurate bbox, can lower via settings for speed)
    dpi = int(getattr(settings, "tesseract_render_dpi", 300))
    if dpi <= 0:
        dpi = 300

    psm_mode = _sanitize_psm_mode(getattr(settings, "tesseract_psm_mode", DEFAULT_PSM_MODE))
    oem_mode = _sanitize_oem_mode(getattr(settings, "tesseract_oem_mode", DEFAULT_OEM_MODE))
    min_confidence = _sanitize_min_confidence(
        getattr(settings, "tesseract_min_confidence", DEFAULT_MIN_CONFIDENCE)
    )
    granularity = _sanitize_granularity(getattr(settings, "tesseract_granularity", "word"))
    extra_config = str(getattr(settings, "tesseract_config_string", "") or "").strip()
    if bool(getattr(settings, "tesseract_disable_dawg", False)):
        # Disable word dictionary bias so misspellings (rates->raates) are less likely to be normalized.
        extra_config = (extra_config + " -c load_system_dawg=0 -c load_freq_dawg=0").strip()
    tesseract_config = _build_tesseract_config(psm_mode, oem_mode, extra_config, dpi=dpi)
    logger.info("[Tesseract] lang=%s", getattr(settings, "tesseract_lang", None))
    logger.info(
        "[Tesseract] Config: PSM=%d OEM=%d granularity=%s min_conf=%d DPI=%d config='%s'",
        psm_mode,
        oem_mode,
        granularity,
        min_confidence,
        dpi,
        tesseract_config,
    )

    def process_page(page_num: int) -> PageData:
        """Process a single page through Tesseract OCR."""
        page_start = time.time()
        page = doc[page_num]
        page_num_display = page_num + 1
        
        # Render page at configured DPI (lower = faster)
        pix = page.get_pixmap(dpi=dpi)

        # Convert pixmap to PIL Image for Tesseract (channel-aware)
        img_raw = _pixmap_to_pil(pix)
        # Keep grayscale version for bbox tightening/debug
        img_gray = img_raw.convert("L")
        # Preprocess for better OCR accuracy (default: mild grayscale preprocessing)
        img = _preprocess_for_tesseract(img_raw)

        def _run_pass(psm_try: int) -> tuple[List[TextBlock], int, str]:
            conf = _build_tesseract_config(psm_try, oem_mode, extra_config, dpi=dpi)
            try:
                ocr_data_local = pytesseract.image_to_data(
                    img,
                    lang=settings.tesseract_lang,
                    output_type=Output.DICT,
                    config=conf,
                )
            except TesseractError as exc:
                logger.warning("[Tesseract] image_to_data failed (PSM=%d): %s", psm_try, exc)
                return [], 0, conf

            blocks_local = _tesseract_data_to_text_blocks(
                ocr_data_local,
                dpi=dpi,
                granularity=granularity,
                min_confidence=min_confidence,
            )

            # Tighten word-level bboxes to actual ink extent
            if granularity == "word" and blocks_local:
                for block in blocks_local:
                    block.bbox = _tighten_bbox_to_ink(img_gray, block.bbox, dpi=dpi, pad_px=1)

            chars_local = sum(len(b.text or "") for b in blocks_local)
            return blocks_local, chars_local, conf

        # OCR: optionally try multiple PSM modes if the first pass returns too little text.
        psm_fallback_enabled = bool(getattr(settings, "tesseract_psm_fallback_enabled", True))
        psm_fallback_order = list(getattr(settings, "tesseract_psm_fallback_order", [6, 4, 3, 11, 12, 1]))
        if psm_mode in psm_fallback_order:
            # Keep user-configured PSM first.
            psm_tries = [psm_mode] + [p for p in psm_fallback_order if p != psm_mode]
        else:
            psm_tries = [psm_mode] + psm_fallback_order

        min_chars_for_psm_ok = int(getattr(settings, "tesseract_psm_min_chars_ok", 200))
        if min_chars_for_psm_ok <= 0:
            min_chars_for_psm_ok = 200

        best_blocks: List[TextBlock] = []
        best_chars = -1
        best_conf_used = tesseract_config

        for idx_try, psm_try in enumerate(psm_tries):
            if idx_try > 0 and not psm_fallback_enabled:
                break
            blocks_try, chars_try, conf_used = _run_pass(int(psm_try))
            logger.info(
                "[Tesseract] Page %d PSM=%d -> %d blocks, %d chars",
                page_num_display,
                int(psm_try),
                len(blocks_try),
                chars_try,
            )
            if chars_try > best_chars:
                best_blocks = blocks_try
                best_chars = chars_try
                best_conf_used = conf_used
            if chars_try >= min_chars_for_psm_ok:
                break

        text_blocks = best_blocks
        
        if granularity == "word" and not text_blocks:
            try:
                boxes_text = pytesseract.image_to_boxes(
                    img,
                    lang=settings.tesseract_lang,
                    config=tesseract_config,
                )
                ocr_text = pytesseract.image_to_string(
                    img,
                    lang=settings.tesseract_lang,
                    config=tesseract_config,
                )
                text_blocks = _tesseract_boxes_to_text_blocks(
                    boxes_text,
                    ocr_text,
                    dpi=dpi,
                    image_height=img.height,
                )
                if text_blocks:
                    logger.info("[Tesseract] Fallback to image_to_boxes produced %d blocks", len(text_blocks))
            except TesseractError as exc:
                logger.warning("[Tesseract] image_to_boxes fallback failed: %s", exc)

        # Debug safeguard: if output is still tiny, save the rendered/preprocessed image.
        try:
            debug_min_chars = int(getattr(settings, "tesseract_debug_min_chars", 50))
        except Exception:
            debug_min_chars = 50
        debug_save_enabled = bool(getattr(settings, "tesseract_debug_save_low_text_images", True))
        if debug_save_enabled:
            page_chars = sum(len(b.text or "") for b in text_blocks)
            if page_chars < debug_min_chars:
                debug_dir = Path(getattr(settings, "debug_output_path", "./debug_output"))
                debug_dir.mkdir(parents=True, exist_ok=True)
                out_path = debug_dir / f"tesseract_lowtext_p{page_num_display}_dpi{dpi}.png"
                try:
                    img.save(out_path)
                    logger.warning(
                        "[Tesseract] Low text (%d chars). Saved debug image: %s",
                        page_chars,
                        out_path,
                    )
                except Exception as exc:
                    logger.debug("[Tesseract] Failed to save debug image: %s", exc)

        page_data = PageData(
            page_num=page_num_display,
            width=page.rect.width,
            height=page.rect.height,
            blocks=text_blocks,
        )
        page_data.metadata = {
            "extraction_method": "ocr_tesseract",
            "ocr_engine_used": "tesseract",
            "dpi": dpi,
            "psm_mode": psm_mode,
            "oem_mode": oem_mode,
            "granularity": granularity,
            "min_confidence": min_confidence,
            "tesseract_config": best_conf_used,
        }

        if text_blocks:
            avg_conf = sum(b.metadata.get("confidence", 0.0) for b in text_blocks) / len(text_blocks)
        else:
            avg_conf = 0.0
        
        page_time = time.time() - page_start
        logger.info(
            "[Tesseract] Page %d/%d: %d blocks, avg_conf=%.2f in %.2fs",
            page_num_display,
            total_pages,
            len(text_blocks),
            avg_conf,
            page_time,
        )
        
        return page_data

    # Process pages - use parallel processing for multi-page documents
    pages: List[PageData] = [None] * total_pages  # Pre-allocate to maintain order
    
    # Determine number of workers (Tesseract can use more since it's lighter per-call)
    num_workers = min(getattr(settings, 'num_workers', 2), 4, total_pages)
    
    if num_workers > 1 and total_pages > 2:
        logger.info("[Tesseract] Using %d parallel workers", num_workers)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_page = {executor.submit(process_page, i): i for i in range(total_pages)}
            for future in as_completed(future_to_page):
                page_idx = future_to_page[future]
                try:
                    pages[page_idx] = future.result()
                except Exception as exc:
                    logger.error("[Tesseract] Page %d failed: %s", page_idx + 1, exc)
                    # Create empty page data on failure
                    page = doc[page_idx]
                    pages[page_idx] = PageData(
                        page_num=page_idx + 1,
                        width=page.rect.width,
                        height=page.rect.height,
                        blocks=[],
                    )
    else:
        # Sequential processing for small documents
        for i in range(total_pages):
            pages[i] = process_page(i)

    doc.close()
    total_time = time.time() - total_start
    logger.info("[Tesseract] OCR processed %d pages in %.2fs (%.2fs/page)", 
               len(pages), total_time, total_time / max(len(pages), 1))
    return pages


def _tesseract_data_to_text_blocks(
    ocr_data: dict,
    *,
    dpi: int,
    granularity: str,
    min_confidence: int,
) -> List[TextBlock]:
    """
    Convert Tesseract OCR data to TextBlock format.

    Tesseract returns dict with keys: 'left', 'top', 'width', 'height', 'text', 'conf', etc.

    We convert to our format: {"x": x, "y": y, "width": w, "height": h}
    """
    text_blocks: List[TextBlock] = []

    if not ocr_data or "text" not in ocr_data:
        return text_blocks

    granularity = _sanitize_granularity(granularity)
    min_confidence = _sanitize_min_confidence(min_confidence)

    n_boxes = len(ocr_data["text"])

    if dpi <= 0:
        dpi = 150
    scale_factor = 72.0 / float(dpi)  # Convert pixels to PDF points

    levels = ocr_data.get("level", [5] * n_boxes)
    block_nums = ocr_data.get("block_num", [0] * n_boxes)
    par_nums = ocr_data.get("par_num", [0] * n_boxes)
    line_nums = ocr_data.get("line_num", [0] * n_boxes)
    word_nums = ocr_data.get("word_num", [0] * n_boxes)

    word_entries = []
    for i in range(n_boxes):
        level = _coerce_int(levels[i], 5) if i < len(levels) else 5
        if level != 5:
            continue
        text = str(ocr_data["text"][i] or "").strip()
        if not text:
            continue
        conf_raw = ocr_data.get("conf", ["0"] * n_boxes)[i]
        conf_i, conf = _parse_confidence(conf_raw)
        low_confidence = conf_i < min_confidence
        bbox = _build_bbox(
            ocr_data.get("left", [0] * n_boxes)[i],
            ocr_data.get("top", [0] * n_boxes)[i],
            ocr_data.get("width", [0] * n_boxes)[i],
            ocr_data.get("height", [0] * n_boxes)[i],
            scale_factor,
        )
        if bbox is None:
            continue
        word_entries.append({
            "text": text,
            "bbox": bbox,
            "conf": conf,
            "conf_raw": conf_i,
            "low_confidence": low_confidence,
            "block_num": _coerce_int(block_nums[i], 0) if i < len(block_nums) else 0,
            "par_num": _coerce_int(par_nums[i], 0) if i < len(par_nums) else 0,
            "line_num": _coerce_int(line_nums[i], 0) if i < len(line_nums) else 0,
            "word_num": _coerce_int(word_nums[i], 0) if i < len(word_nums) else 0,
            "order": i,
        })

    if granularity == "word":
        for entry in word_entries:
            block = TextBlock(
                text=entry["text"],
                bbox=entry["bbox"],
                style=None,
                metadata={
                    "granularity": "word",
                    "level": "word",
                    "ocr_engine": "tesseract",
                    "bbox_source": "exact",
                    "bbox_units": "pt",
                    "bbox_space": "page",
                    "confidence": entry["conf"],
                    "confidence_raw": entry.get("conf_raw", 0),
                    "low_confidence": bool(entry.get("low_confidence", False)),
                    "text_source": "tesseract_ocr",
                },
            )
            text_blocks.append(block)
        return text_blocks

    grouped = _group_word_entries(word_entries, granularity)
    for group in grouped:
        # Do not drop low-confidence groups; keep them and mark for downstream weighting.
        group_low_conf = group["confidence"] < (min_confidence / 100.0)
        block = TextBlock(
            text=group["text"],
            bbox=group["bbox"],
            style=None,
            metadata={
                "granularity": granularity,
                "level": granularity,
                "ocr_engine": "tesseract",
                "bbox_source": "exact",
                "bbox_units": "pt",
                "bbox_space": "page",
                "confidence": group["confidence"],
                "low_confidence": bool(group_low_conf),
                "text_source": "tesseract_ocr",
                "words": group["words"],
            },
        )
        text_blocks.append(block)

    return text_blocks
