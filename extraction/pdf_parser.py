"""Digital PDF text/style extraction using PyMuPDF."""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from comparison.models import PageData, Style, TextBlock
from extraction.layout_analyzer import analyze_layout
from utils.coordinates import bbox_tuple_to_dict, normalize_page_bboxes
from utils.logging import logger

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None


def parse_pdf_words_as_lines(
    path: str | Path, *, run_layout_analysis: bool = True
) -> List[PageData]:
    """
    Digital PDF extractor that returns canonical LINE blocks,
    each carrying WORD boxes in metadata["words"].

    This provides word-level bbox granularity for precise diff highlighting.

    Why:
    - page.get_text("dict") blocks are paragraph-ish → too coarse
    - page.get_text("words") gives true word bboxes → perfect canonical base
    """
    path = Path(path)
    logger.info("Parsing PDF (digital/words->lines): %s", path)

    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required for PDF parsing. Install via `pip install PyMuPDF`."
        )

    pages: List[PageData] = []

    with fitz.open(path) as doc:
        for page in doc:
            page_w, page_h = page.rect.width, page.rect.height
            page_data = PageData(page_num=page.number + 1, width=page_w, height=page_h)

            # OPTIMIZATION: Extract text dict once per page for style lookup
            # This avoids calling page.get_text("dict") for every single line
            flags = _get_text_flags(fitz)
            try:
                # sort=True stabilizes reading order across PDFs where internal object order differs.
                text_dict = page.get_text("dict", flags=flags, sort=True)
            except TypeError:
                # Backward compatibility: older PyMuPDF versions may not accept sort/flags combos.
                try:
                    text_dict = page.get_text("dict", flags=flags)
                except TypeError:
                    text_dict = page.get_text("dict")

            # Build a flat span index for word-level style lookup.
            # Each span includes an exact bbox and style attributes.
            span_index = _build_span_index(text_dict)

            cached_lines = []
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []) or []:
                    bbox = line.get("bbox")
                    if bbox and len(bbox) >= 4:
                        # Store (x0, y0, x1, y1, line_dict)
                        cached_lines.append((bbox[0], bbox[1], bbox[2], bbox[3], line))
            
            # Sort by y0 to potentially speed up search (though linear scan is fast enough for pages)
            cached_lines.sort(key=lambda x: x[1])

            # words: (x0,y0,x1,y1, word, block_no, line_no, word_no)
            try:
                words = page.get_text("words", flags=flags, sort=True)
            except TypeError:
                # Backward compatibility: some PyMuPDF versions don't accept flags/sort for "words".
                try:
                    words = page.get_text("words", flags=flags)
                except TypeError:
                    try:
                        words = page.get_text("words", sort=True)
                    except TypeError:
                        words = page.get_text("words")

            # Group by (block_no, line_no) => canonical LINE
            grouped: Dict[Tuple[int, int], List[tuple]] = {}
            for w in words:
                x0, y0, x1, y1, txt, block_no, line_no, word_no = w
                txt = unicodedata.normalize("NFC", (txt or "").strip())
                if not txt:
                    continue
                grouped.setdefault((int(block_no), int(line_no)), []).append(w)

            # Stable reading order: sort by min_y then min_x
            def line_sort_key(items: List[tuple]):
                xs0 = [it[0] for it in items]
                ys0 = [it[1] for it in items]
                return (min(ys0), min(xs0))

            line_items = sorted(grouped.values(), key=line_sort_key)

            for items in line_items:
                # Sort words in line by word_no then x0
                items = sorted(items, key=lambda it: (int(it[7]), float(it[0])))
                line_text = " ".join(
                    unicodedata.normalize("NFC", (it[4] or "").strip()) for it in items
                ).strip()
                if not line_text:
                    continue

                # Union bbox for the line
                x0 = min(float(it[0]) for it in items)
                y0 = min(float(it[1]) for it in items)
                x1 = max(float(it[2]) for it in items)
                y1 = max(float(it[3]) for it in items)

                words_meta = []
                for it in items:
                    wx0, wy0, wx1, wy1, wtxt, *_ = it
                    wtxt_norm = unicodedata.normalize("NFC", ("" if wtxt is None else str(wtxt)).strip())
                    word_style = _find_best_span_style(span_index, float(wx0), float(wy0), float(wx1), float(wy1))
                    words_meta.append({
                        "text": wtxt_norm,
                        "bbox": bbox_tuple_to_dict(
                            (float(wx0), float(wy0), float(wx1), float(wy1))
                        ),
                        "conf": 1.0,
                        "style": _style_to_dict(word_style) if word_style else None,
                    })

                # Extract a representative line style (best-overlapping span in best-overlapping line)
                style = _find_style_for_line_in_cache(cached_lines, x0, y0, x1, y1)

                page_data.blocks.append(
                    TextBlock(
                        text=line_text,
                        bbox=bbox_tuple_to_dict((x0, y0, x1, y1)),
                        style=style,
                        metadata={
                            "granularity": "line",
                            "bbox_units": "pt",
                            "bbox_source": "pymupdf_words",
                            "text_source": "pymupdf_words",
                            "words": words_meta,  # canonical word layer inside every line
                        },
                    )
                )

            page_data.metadata = {
                "rotation": page.rotation,
                "cropbox": list(page.cropbox) if page.cropbox else None,
                "extraction_method": "pdf_digital_words",
                "ocr_engine_used": None,
                "pymupdf_text_flags": int(flags),
                # Used by downstream components (e.g., figure visual hashing)
                # to re-open the original PDF and render clipped regions.
                "source_pdf_path": str(path),
                "page_index": int(page.number),
            }

            pages.append(page_data)

    # Run layout analysis if requested
    if run_layout_analysis and path.exists():
        layout_pages = analyze_layout(path)
        layout_lookup = {p.page_num: p for p in layout_pages}

        for page_data in pages:
            if page_data.page_num in layout_lookup:
                layout_page = layout_lookup[page_data.page_num]
                page_data.metadata.update(layout_page.metadata)
            # Preserve source pointers even if layout metadata overwrote keys.
            page_data.metadata.setdefault("source_pdf_path", str(path))
            page_data.metadata.setdefault("page_index", int(page_data.page_num - 1))
    else:
        for page_data in pages:
            page_data.metadata.setdefault("layout_analyzed", False)

    pages = normalize_page_bboxes(pages)
    logger.info("Extracted %d pages with word-level granularity from PDF", len(pages))
    return pages


def _find_style_for_line_in_cache(
    cached_lines: List[tuple],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> Style:
    """
    Find matching line in cache and extract style.
    
    Args:
        cached_lines: List of (x0, y0, x1, y1, line_dict) tuples
        x0, y0, x1, y1: Bounds of the target line
    """
    target = (float(x0), float(y0), float(x1), float(y1))

    best_line = None
    best_line_overlap = 0.0
    for lx0, ly0, lx1, ly1, line in cached_lines:
        cand = (float(lx0), float(ly0), float(lx1), float(ly1))
        ov = _bbox_intersection_area(target, cand)
        if ov > best_line_overlap:
            best_line_overlap = ov
            best_line = line

    if best_line:
        # Within the chosen line, pick the span that overlaps the target bbox the most.
        best_span = None
        best_span_overlap = 0.0
        for span in best_line.get("spans", []) or []:
            sb = span.get("bbox")
            if not sb or len(sb) != 4:
                continue
            span_bbox = (float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3]))
            ov = _bbox_intersection_area(target, span_bbox)
            if ov > best_span_overlap:
                best_span_overlap = ov
                best_span = span

        if best_span:
            return _extract_style_from_span(best_span)

    return Style(font=None, size=None, bold=False, italic=False, color=None)


def _build_span_index(text_dict: dict) -> List[dict]:
    """Flatten PyMuPDF dict spans into a list for fast overlap lookup."""
    out: List[dict] = []
    for block in text_dict.get("blocks", []) or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                sb = span.get("bbox")
                if not sb or len(sb) != 4:
                    continue
                try:
                    bbox = (float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3]))
                except Exception:
                    continue
                out.append({
                    "bbox": bbox,
                    "style": _extract_style_from_span(span),
                })
    return out


def _find_best_span_style(
    span_index: List[dict],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> Optional[Style]:
    """Pick the style of the span with maximum bbox overlap with the word bbox."""
    target = (float(x0), float(y0), float(x1), float(y1))
    best = None
    best_ov = 0.0
    for item in span_index:
        sb = item.get("bbox")
        if not sb:
            continue
        # Cheap Y pruning first
        if sb[1] >= target[3] or sb[3] <= target[1]:
            continue
        ov = _bbox_intersection_area(target, sb)
        if ov > best_ov:
            best_ov = ov
            best = item
    if not best:
        return None
    st = best.get("style")
    return st if isinstance(st, Style) else None


def _bbox_intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Intersection area between two (x0,y0,x1,y1) bboxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = ix1 - ix0
    ih = iy1 - iy0
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    return iw * ih


def _style_to_dict(style: Style) -> dict:
    """Serialize Style for word metadata (JSON-friendly, stable-ish)."""
    size = style.size
    try:
        size_out = None if size is None else round(float(size), 2)
    except Exception:
        size_out = None
    return {
        "font": style.font,
        "size": size_out,
        "bold": bool(style.bold),
        "italic": bool(style.italic),
        "color": list(style.color) if isinstance(style.color, tuple) else None,
    }


def _extract_style_from_span(span: dict) -> Style:
    """Extract style from a single span."""
    font_name = span.get("font")
    
    # Variant B: Normalize font name to handle subsets
    # 1. ABCDEF+Calibri -> Calibri (standard subset)
    # 2. CIDFont+F1 -> CIDFont (generic stable name)
    if font_name and "+" in font_name:
        prefix, suffix = font_name.split("+", 1)
        if prefix == "CIDFont":
            font_name = "CIDFont"
        elif len(prefix) == 6 and prefix.isupper():
            font_name = suffix

    font_size = span.get("size")
    flags = int(span.get("flags", 0) or 0)
    is_bold = bool(flags & 16)
    is_italic = bool(flags & 2)
    
    color = None
    color_val = span.get("color")
    if color_val is not None and isinstance(color_val, int):
        r = (color_val >> 16) & 0xFF
        g = (color_val >> 8) & 0xFF
        b = color_val & 0xFF
        color = (r, g, b)
    elif isinstance(color_val, (tuple, list)) and len(color_val) == 3:
        try:
            color = (int(color_val[0]), int(color_val[1]), int(color_val[2]))
        except Exception:
            color = None
    
    return Style(
        font=font_name,
        size=font_size,
        bold=is_bold,
        italic=is_italic,
        color=color,
    )


def parse_pdf(path: str | Path, *, run_layout_analysis: bool = True) -> List[PageData]:
    """
    Extract text, fonts, and positions from a digital PDF using PyMuPDF.

    Performance note (important):
    - We use a **single** `page.get_text("dict", ...)` call per page and derive both
      text blocks and style from that structure. This avoids repeatedly calling
      `get_text()` (which is expensive) for each block.
    """
    path = Path(path)
    logger.info("Parsing PDF (digital): %s", path)
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required for PDF parsing. Install via `pip install PyMuPDF`."
        )
    
    pages: List[PageData] = []

    flags = _get_text_flags(fitz)

    # Use context manager to ensure file handles are closed promptly.
    with fitz.open(path) as doc:
        for page in doc:
            width, height = page.rect.width, page.rect.height
            page_data = PageData(page_num=page.number + 1, width=width, height=height)

            # Extract text blocks and style in one pass.
            text_dict = page.get_text("dict", flags=flags)
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:  # not a text block
                    continue

                bbox = block.get("bbox", None)
                if not bbox or len(bbox) != 4:
                    continue

                x0, y0, x1, y1 = bbox
                bbox_dict = bbox_tuple_to_dict((x0, y0, x1, y1))

                # Build block text from spans.
                text = _extract_block_text(block)
                if not text.strip():
                    continue

                style = _extract_style_from_block(block)
                page_data.blocks.append(
                    TextBlock(
                        text=text.strip(),
                        bbox=bbox_dict,
                        style=style,
                        metadata={
                            "bbox_units": "pt",
                            "bbox_source": "exact",
                            "bbox_space": "page",
                            "text_source": "pymupdf_dict",
                        },
                    )
                )

            # Store page metadata
            page_data.metadata = {
                "rotation": page.rotation,
                "cropbox": list(page.cropbox) if page.cropbox else None,
                "extraction_method": "pdf_digital",
                "ocr_engine_used": None,
                "pymupdf_text_flags": int(flags),
            }

            pages.append(page_data)
    
    if run_layout_analysis and path.exists():
        # Run layout analysis to detect tables and figures
        layout_pages = analyze_layout(path)
        layout_lookup = {p.page_num: p for p in layout_pages}

        # Merge layout metadata into extracted pages
        for page_data in pages:
            if page_data.page_num in layout_lookup:
                layout_page = layout_lookup[page_data.page_num]
                page_data.metadata.update(layout_page.metadata)
    else:
        # Keep explicit signal so downstream code can skip layout-dependent logic.
        for page_data in pages:
            page_data.metadata.setdefault("layout_analyzed", False)
    
    pages = normalize_page_bboxes(pages)  # bbox police checkpoint ✅
    logger.info("Extracted %d pages from PDF", len(pages))
    return pages


def _get_text_flags(fitz_module) -> int:
    """
    Build a robust set of PyMuPDF text extraction flags.

    We prefer preserving ligatures and whitespace for better downstream diffs,
    while remaining compatible across PyMuPDF versions.
    """
    flags = 0
    flags |= int(getattr(fitz_module, "TEXT_PRESERVE_LIGATURES", 0))
    flags |= int(getattr(fitz_module, "TEXT_PRESERVE_WHITESPACE", 0))
    # Preserve images is not needed for text extraction; keep minimal flags to reduce work.
    return flags


def _extract_block_text(block: dict) -> str:
    """Extract text for a PyMuPDF dict 'block' by concatenating spans line by line."""
    lines_out: List[str] = []
    for line in block.get("lines", []) or []:
        spans = line.get("spans", []) or []
        if not spans:
            continue
        line_text = "".join((s.get("text") or "") for s in spans)
        if line_text:
            lines_out.append(line_text)
    return "\n".join(lines_out)


def _extract_style_from_block(block: dict) -> Style:
    """
    Extract a representative style from a PyMuPDF dict block.

    We take the first span we see as the "representative" style. This matches the
    previous behavior (first overlapping span) but avoids repeated get_text calls.
    """
    font_name = None
    font_size = None
    is_bold = False
    is_italic = False
    color = None

    for line in block.get("lines", []) or []:
        for span in line.get("spans", []) or []:
            # Use first span's style as representative
            if font_name is None:
                font_name = span.get("font")
            if font_size is None:
                font_size = span.get("size")
            
            flags = int(span.get("flags", 0) or 0)
            if not is_bold:
                is_bold = bool(flags & 16)  # Bit 4 indicates bold
            if not is_italic:
                is_italic = bool(flags & 2)  # Bit 1 indicates italic
            
            if color is None:
                color_val = span.get("color")
                if color_val is not None and isinstance(color_val, int):
                    r = (color_val >> 16) & 0xFF
                    g = (color_val >> 8) & 0xFF
                    b = color_val & 0xFF
                    color = (r, g, b)
                elif isinstance(color_val, (tuple, list)) and len(color_val) == 3:
                    try:
                        color = (int(color_val[0]), int(color_val[1]), int(color_val[2]))
                    except Exception:
                        color = None
            
            # Once we have all style info from first span, return early
            if font_name and font_size is not None:
                break
        
        if font_name and font_size is not None:
            break

    return Style(
        font=font_name,
        size=font_size,
        bold=is_bold,
        italic=is_italic,
        color=color,
    )
