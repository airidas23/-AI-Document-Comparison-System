"""Line-level extraction with canonical line_id generation."""
from __future__ import annotations

import hashlib
import re
import statistics
from pathlib import Path
from typing import List, Optional

from comparison.models import Line, PageData, Token
from config.settings import settings
from utils.coordinates import bbox_tuple_to_dict, normalize_bbox
from utils.logging import logger

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None


def normalize_text_for_id(text: str) -> str:
    """Normalize text for stable line_id generation."""
    if not text:
        return ""
    lowered = text.lower()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def quantize(value: float, bin_size: float = 0.02) -> float:
    """Quantize a normalized value into bins (default 2% bins)."""
    if bin_size <= 0:
        return value
    return round(value / bin_size) * bin_size


def bbox_union(bboxes: List[dict]) -> dict:
    """Merge multiple bbox dicts into their union."""
    if not bboxes:
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    x_min = min(b["x"] for b in bboxes)
    y_min = min(b["y"] for b in bboxes)
    x_max = max(b["x"] + b["width"] for b in bboxes)
    y_max = max(b["y"] + b["height"] for b in bboxes)
    return {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min}


def make_line_id(
    text: str,
    bbox: dict,
    page_num: int,
    page_width: float,
    page_height: float,
) -> str:
    """Create a stable line_id from normalized text + quantized position + page index."""
    normalized_text = normalize_text_for_id(text) or "empty"
    normalized_bbox = normalize_bbox(
        (bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
        page_width,
        page_height,
    )
    center_x = normalized_bbox["x"] + normalized_bbox["width"] / 2
    center_y = normalized_bbox["y"] + normalized_bbox["height"] / 2
    qx = quantize(center_x)
    qy = quantize(center_y)
    seed = f"{page_num}|{normalized_text}|{qx:.2f}|{qy:.2f}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def group_tokens_into_lines(
    tokens: List[Token],
    *,
    y_threshold: Optional[float] = None,
) -> List[List[Token]]:
    """Group tokens into lines using adaptive Y-proximity."""
    if not tokens:
        return []

    token_info = []
    for token in tokens:
        bbox = token.bbox
        x_center = bbox["x"] + bbox["width"] / 2
        y_center = bbox["y"] + bbox["height"] / 2
        token_info.append((token, x_center, y_center, bbox["height"]))

    token_info.sort(key=lambda item: (item[2], item[1]))

    if y_threshold is None:
        heights = [h for _, _, _, h in token_info if h > 0]
        median_height = statistics.median(heights) if heights else 0.0
        y_threshold = max(2.0, median_height * 0.6)

    lines: List[List[Token]] = []
    current_line: List[Token] = []
    current_y: Optional[float] = None

    for token, _, y_center, _ in token_info:
        if current_y is None:
            current_line = [token]
            current_y = y_center
            continue

        if abs(y_center - current_y) <= y_threshold:
            current_line.append(token)
            current_y = (current_y * (len(current_line) - 1) + y_center) / len(current_line)
        else:
            lines.append(current_line)
            current_line = [token]
            current_y = y_center

    if current_line:
        lines.append(current_line)

    return lines


def _line_center(bbox: dict) -> tuple[float, float]:
    return (bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)


def _build_lines_from_token_groups(
    token_groups: List[List[Token]],
    page_num: int,
    page_width: float,
    page_height: float,
) -> List[Line]:
    lines: List[Line] = []

    for group in token_groups:
        if not group:
            continue
        tokens_sorted = sorted(group, key=lambda t: t.bbox["x"])
        text = " ".join(token.text for token in tokens_sorted).strip()
        if not text:
            continue
        line_bbox = bbox_union([token.bbox for token in tokens_sorted])
        avg_conf = sum(t.confidence for t in tokens_sorted) / len(tokens_sorted)
        line_id = make_line_id(text, line_bbox, page_num, page_width, page_height)
        lines.append(
            Line(
                line_id=line_id,
                bbox=line_bbox,
                text=text,
                confidence=avg_conf,
                tokens=tokens_sorted,
            )
        )

    lines.sort(key=lambda line: (_line_center(line.bbox)[1], _line_center(line.bbox)[0]))
    for idx, line in enumerate(lines):
        line.reading_order = idx

    return lines


def extract_digital_lines(path: str | Path) -> List[PageData]:
    """Extract line-level data from digital PDFs using PyMuPDF words."""
    path = Path(path)
    if fitz is None:
        logger.warning("PyMuPDF not available; cannot extract digital lines.")
        return []

    pages: List[PageData] = []
    with fitz.open(path) as doc:
        for page in doc:
            page_num = page.number + 1
            width, height = page.rect.width, page.rect.height
            tokens: List[Token] = []

            words = page.get_text("words") or []
            for idx, word in enumerate(words):
                if len(word) < 5:
                    continue
                x0, y0, x1, y1, text = word[:5]
                if not text or not str(text).strip():
                    continue
                tokens.append(
                    Token(
                        token_id=f"p{page_num}_w{idx + 1}",
                        bbox=bbox_tuple_to_dict((x0, y0, x1, y1)),
                        text=str(text),
                        confidence=1.0,
                    )
                )

            token_groups = group_tokens_into_lines(tokens)
            lines = _build_lines_from_token_groups(token_groups, page_num, width, height)
            page_data = PageData(page_num=page_num, width=width, height=height, lines=lines)
            page_data.metadata = {
                "line_extraction_method": "pdf_digital",
                "line_extraction_engine": "pymupdf_words",
            }
            pages.append(page_data)

    logger.info("Digital line extraction: %d pages", len(pages))
    return pages


def _paddle_results_to_tokens(
    ocr_result: List,
    *,
    dpi: int,
    page_num: int,
) -> List[Token]:
    tokens: List[Token] = []
    if not ocr_result:
        return tokens

    if dpi <= 0:
        dpi = 150
    scale_factor = 72.0 / float(dpi)

    token_idx = 0
    for result in ocr_result:
        if not isinstance(result, dict):
            continue
        rec_texts = result.get("rec_texts", [])
        rec_scores = result.get("rec_scores", [])
        dt_polys = result.get("dt_polys", [])

        for i, text in enumerate(rec_texts):
            if not text or not str(text).strip():
                continue
            confidence = rec_scores[i] if i < len(rec_scores) else 1.0
            try:
                confidence_f = float(confidence)
            except Exception:
                confidence_f = 0.0

            if confidence_f < float(getattr(settings, "paddle_text_rec_score_thresh", 0.0)):
                continue

            if i >= len(dt_polys):
                continue
            polygon = dt_polys[i]
            x_coords = [point[0] for point in polygon]
            y_coords = [point[1] for point in polygon]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            bbox = {
                "x": float(x_min) * scale_factor,
                "y": float(y_min) * scale_factor,
                "width": float(x_max - x_min) * scale_factor,
                "height": float(y_max - y_min) * scale_factor,
            }

            token_idx += 1
            tokens.append(
                Token(
                    token_id=f"p{page_num}_ocr{token_idx}",
                    bbox=bbox,
                    text=str(text),
                    confidence=confidence_f,
                )
            )

    return tokens


def extract_ocr_lines(path: str | Path) -> List[PageData]:
    """Extract line-level data from scanned PDFs using PaddleOCR."""
    path = Path(path)
    if fitz is None:
        logger.warning("PyMuPDF not available; cannot render for OCR line extraction.")
        return []

    from extraction.paddle_ocr_engine import _get_ocr

    dpi = int(getattr(settings, "paddle_render_dpi", 150))
    if dpi <= 0:
        dpi = 150

    ocr = _get_ocr()
    pages: List[PageData] = []

    with fitz.open(path) as doc:
        for page in doc:
            page_num = page.number + 1
            width, height = page.rect.width, page.rect.height

            pix = page.get_pixmap(dpi=dpi)
            from PIL import Image
            import numpy as np

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_array = np.array(img)

            ocr_result = ocr.predict(img_array)
            tokens = _paddle_results_to_tokens(ocr_result, dpi=dpi, page_num=page_num)
            token_groups = group_tokens_into_lines(tokens)
            lines = _build_lines_from_token_groups(token_groups, page_num, width, height)

            page_data = PageData(page_num=page_num, width=width, height=height, lines=lines)
            page_data.metadata = {
                "line_extraction_method": "ocr_paddle",
                "line_extraction_engine": "paddle",
                "dpi": dpi,
            }
            pages.append(page_data)

    logger.info("OCR line extraction: %d pages", len(pages))
    return pages


def extract_lines_from_pages(pages: List[PageData]) -> List[PageData]:
    """
    Convert existing blocks to lines WITHOUT re-running OCR.
    
    This function takes already-extracted PageData (with blocks from OCR)
    and converts them to line-level format. This avoids calling OCR twice.
    
    Args:
        pages: List of PageData with populated blocks (from extract_pdf/OCR)
    
    Returns:
        Same pages with lines populated from blocks
    """

    def _approx_word_tokens_from_block(text: str, bbox: dict) -> List[Token]:
        """Fallback: approximate whitespace-delimited word tokens inside a block bbox.

        This is used when OCR engines provide only line-level bboxes.
        """
        if not text or not str(text).strip():
            return []
        if not bbox or bbox.get("width", 0.0) <= 0.0 or bbox.get("height", 0.0) <= 0.0:
            return []

        # Split on whitespace (better match for visual word spacing than punctuation tokenization).
        words = [w for w in re.split(r"\s+", str(text).strip()) if w]
        if len(words) <= 1:
            return []

        x0 = float(bbox.get("x", 0.0))
        y0 = float(bbox.get("y", 0.0))
        w = float(bbox.get("width", 0.0))
        h = float(bbox.get("height", 0.0))

        # Allocate horizontal space proportionally to word length.
        # Add small weight for spaces so boundaries stay stable.
        space_weight = 1.0
        weights = [max(1.0, float(len(wd))) for wd in words]
        total = sum(weights) + space_weight * (len(words) - 1)

        cur = x0
        out: List[Token] = []
        for i, (wd, wt) in enumerate(zip(words, weights)):
            if i > 0:
                cur += w * (space_weight / total)
            ww = w * (wt / total)
            out.append(
                Token(
                    token_id="",
                    bbox={"x": cur, "y": y0, "width": max(0.0, ww), "height": max(0.0, h)},
                    text=str(wd),
                    confidence=1.0,
                )
            )
            cur += ww
        return out

    for page in pages:
        if page.lines:
            # Lines already exist, skip
            continue
        
        if not page.blocks:
            page.lines = []
            continue
        
        # Convert blocks to tokens.
        # IMPORTANT: If OCR provides word-level bbox metadata (metadata['words']),
        # use it to build word tokens so content diffs can highlight changed words
        # precisely (Tesseract-like UX).
        tokens: List[Token] = []
        token_seq = 0
        for idx, block in enumerate(page.blocks):
            if not block.text or not str(block.text).strip():
                continue

            md = block.metadata or {}
            blk_conf = md.get("confidence", 1.0)
            try:
                blk_conf = float(blk_conf)
            except (TypeError, ValueError):
                blk_conf = 1.0

            words = md.get("words")
            if isinstance(words, list) and len(words) > 0:
                # Word-level tokens
                for w in words:
                    if not isinstance(w, dict):
                        continue
                    wt = (w.get("text") or "").strip()
                    wb = w.get("bbox")
                    if not wt or not isinstance(wb, dict):
                        continue
                    try:
                        wc = float(w.get("conf", blk_conf))
                    except (TypeError, ValueError):
                        wc = blk_conf
                    token_seq += 1
                    tokens.append(
                        Token(
                            token_id=f"p{page.page_num}_w{token_seq}",
                            bbox=wb.copy(),
                            text=wt,
                            confidence=wc,
                        )
                    )
                continue

            # Fallback: approximate whitespace word tokens when block bbox is present.
            approx = _approx_word_tokens_from_block(block.text, block.bbox)
            if approx:
                for t in approx:
                    token_seq += 1
                    t.token_id = f"p{page.page_num}_aw{token_seq}"
                    t.confidence = blk_conf
                    tokens.append(t)
                continue

            # Last resort: block-level token.
            token_seq += 1
            tokens.append(
                Token(
                    token_id=f"p{page.page_num}_b{idx + 1}",
                    bbox=block.bbox.copy() if isinstance(block.bbox, dict) else block.bbox,
                    text=str(block.text),
                    confidence=blk_conf,
                )
            )
        
        # Group tokens into lines
        token_groups = group_tokens_into_lines(tokens)
        lines = _build_lines_from_token_groups(
            token_groups, page.page_num, page.width, page.height
        )
        page.lines = lines
        
        # Mark metadata
        if not page.metadata:
            page.metadata = {}
        page.metadata["line_extraction_method"] = "from_blocks"
        page.metadata["line_extraction_engine"] = page.metadata.get("ocr_engine_used", "unknown")
    
    logger.info("Converted blocks to lines for %d pages (no additional OCR)", len(pages))
    return pages


def extract_lines(pages: List[PageData]) -> List[PageData]:
    """
    Extract lines from pages - REUSES existing blocks if available.
    
    This is the preferred entry point that avoids double OCR.
    If pages already have blocks (from OCR), converts them to lines.
    Otherwise falls back to digital extraction.
    """
    # Check if pages have blocks (from OCR)
    total_blocks = sum(len(p.blocks) for p in pages)
    if total_blocks > 0:
        logger.info("Reusing %d blocks from pages to create lines (no additional OCR)", total_blocks)
        return extract_lines_from_pages(pages)
    
    # No blocks - try digital extraction from first page's source
    # This is a fallback for when pages were created without OCR
    logger.info("No blocks in pages, lines not extracted")
    return pages


def extract_document_lines(path: str | Path) -> List[PageData]:
    """Extract line-level data with digital-first fallback to OCR."""
    path = Path(path)
    try:
        digital_pages = extract_digital_lines(path)
        digital_line_count = sum(len(p.lines) for p in digital_pages)
        if digital_line_count > 0:
            return digital_pages
        logger.info("Digital line extraction empty; falling back to OCR.")
    except Exception as exc:
        logger.warning("Digital line extraction failed: %s. Falling back to OCR.", exc)

    try:
        return extract_ocr_lines(path)
    except Exception as exc:
        logger.warning("OCR line extraction failed: %s", exc)
        return []
