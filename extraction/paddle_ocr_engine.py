"""PaddleOCR engine for OCR processing with natural bbox support."""
from __future__ import annotations

from pathlib import Path
import threading
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

from comparison.models import PageData, TextBlock
from config.settings import settings
from utils.logging import logger

import re


def _approx_word_boxes(line_text: str, line_bbox: dict) -> list:
    """Approximate per-token bboxes inside a line bbox.

    PaddleOCR returns line-level boxes by default. To enable word-level highlighting
    we split text into tokens and allocate horizontal space by token length.
    This is a heuristic (OCR does not guarantee monospace), but it's stable and consistent.
    """
    if not line_text or not line_bbox:
        return []
    x0 = float(line_bbox.get("x", 0.0))
    y0 = float(line_bbox.get("y", 0.0))
    w = float(line_bbox.get("width", 0.0))
    h = float(line_bbox.get("height", 0.0))
    if w <= 0 or h <= 0:
        return []

    tokens = re.findall(r"\w+|[^\w\s]", line_text, flags=re.UNICODE)
    if not tokens:
        return []
    # Weight tokens by length; add a small weight for spaces between tokens
    space_weight = 1.0
    weights = [max(1.0, float(len(t))) for t in tokens]
    total = sum(weights) + space_weight * (len(tokens) - 1)

    # Minimum visible token size: tie to line height so narrow table cells
    # still produce a usable highlight rectangle.
    min_token_w = max(0.6 * h, 0.01 * w)
    pad_x = max(0.15 * h, 0.0025 * w)
    pad_y = 0.08 * h

    cur = x0
    out = []
    for i, (tok, wt) in enumerate(zip(tokens, weights)):
        if i > 0:
            cur += w * (space_weight / total)
        tw = w * (wt / total)

        # Expand tiny token boxes to avoid dot/circle highlights in the UI.
        # Keep center stable and allow small overlaps between neighbors.
        cx = cur + tw / 2.0
        target_w = max(tw, min_token_w)
        left = cx - target_w / 2.0
        right = cx + target_w / 2.0

        # Add a small padding for visibility.
        left -= pad_x
        right += pad_x
        top = y0 - pad_y
        bottom = y0 + h + pad_y

        # Clamp to line bbox bounds.
        left = max(x0, left)
        right = min(x0 + w, right)
        top = max(y0, top)
        bottom = min(y0 + h, bottom)

        out.append({
            "text": tok,
            "bbox": {"x": left, "y": top, "width": max(0.0, right - left), "height": max(0.0, bottom - top)},
            "conf": 1.0,  # Approximated, no per-word confidence
        })
        cur += tw
    return out


def _group_lines_from_results(
    ocr_result: List,
    scale_factor: float,
) -> List[dict]:
    """
    Group OCR results into line-level structures with word metadata.
    
    Returns list of dicts with:
    - text: full line text
    - bbox: line bounding box
    - words: list of word dicts with text and bbox
    - confidence: average confidence
    """
    lines = []
    
    if not ocr_result:
        return lines
    
    for result in ocr_result:
        if not isinstance(result, dict):
            continue
            
        # Skip structural regions that act as containers (if labeled)
        # We rely on their children (cells/lines) being present or extracted separately
        if result.get('type') in ('table', 'figure'):
            continue
        
        rec_texts = result.get('rec_texts', [])
        rec_scores = result.get('rec_scores', [])
        dt_polys = result.get('dt_polys', [])
        
        # Each detected region becomes a line
        for i, text in enumerate(rec_texts):
            if not text or not text.strip():
                continue
            
            text = text.strip()
            confidence = rec_scores[i] if i < len(rec_scores) else 1.0
            
            # Get polygon coordinates and convert to bbox
            if i < len(dt_polys):
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
            else:
                # Without a polygon we cannot place this line reliably.
                # Skipping avoids generating (0,0,0,0) bboxes that render as dots.
                continue
            
            # Skip degenerate bboxes
            if bbox.get("width", 0.0) <= 0.0 or bbox.get("height", 0.0) <= 0.0:
                continue

            # Generate word-level boxes (approximated)
            words = _approx_word_boxes(text, bbox)
            
            lines.append({
                "text": text,
                "bbox": bbox,
                "words": words,
                "confidence": float(confidence),
            })
    
    return lines

# Module-level cache for PaddleOCR instance
_ocr_instance: Optional[PaddleOCR] = None

# Ensure only one PaddleOCR instance is ever created per process.
# Any concurrent callers will block on this lock.
_init_lock = threading.Lock()

# Optional warmup coordination (useful for background initialization).
_warmup_done = threading.Event()
_warmup_error: Exception | None = None


def warmup_paddle_ocr(*, background: bool = True) -> None:
    """Warm up PaddleOCR by initializing the singleton instance.

    If background=True, starts a daemon thread and returns immediately.
    If background=False, blocks until initialization completes.
    """

    def _do():
        try:
            _get_ocr()
        except Exception:
            # Error is recorded by _get_ocr(); keep the warmup thread silent.
            return

    if background:
        threading.Thread(target=_do, daemon=True, name="PaddleOCRWarmup").start()
    else:
        _do()


def wait_for_paddle_ocr_warmup(timeout: float | None = None) -> bool:
    """Wait for PaddleOCR warmup completion.

    Returns True if the instance is ready.
    Raises RuntimeError if warmup finished but initialization failed.
    Returns False on timeout.
    """
    if _ocr_instance is not None:
        return True

    finished = _warmup_done.wait(timeout=timeout)
    if not finished:
        return False

    if _ocr_instance is not None:
        return True

    if _warmup_error is not None:
        raise RuntimeError("PaddleOCR warmup failed") from _warmup_error

    # Warmup thread ran, but instance wasn't created (shouldn't happen, but be safe)
    return False


def _get_ocr():
    """
    Get or create a cached PaddleOCR instance.
    Caching avoids slow model reloading on every call.
    """
    global _ocr_instance
    global _warmup_error
    
    if _ocr_instance is not None:
        logger.debug("[PaddleOCR] Using cached instance")
        return _ocr_instance

    # Ensure only one thread performs initialization.
    with _init_lock:
        if _ocr_instance is not None:
            logger.debug("[PaddleOCR] Using cached instance")
            return _ocr_instance

        import time
        init_start = time.time()
        logger.info("[PaddleOCR] Initializing PaddleOCR (first time, may take 1-2 minutes)...")
        logger.info("[PaddleOCR] Language: %s", settings.paddle_ocr_lang)
        
        logger.debug("[PaddleOCR] Importing paddleocr module...")
        import_start = time.time()
        from paddleocr import PaddleOCR
        logger.debug("[PaddleOCR] Import took %.2fs", time.time() - import_start)
        
        # Initialize PaddleOCR (v3.x API) 
        # Disable optional models for faster initialization
        
        # Suppress PaddleOCR verbose logging
        import logging
        logging.getLogger("ppocr").setLevel(logging.WARNING)
        
        # Detect GPU availability (for logging purposes - PaddleOCR auto-detects)
        use_gpu = settings.use_gpu
        gpu_available = False
        if use_gpu:
            try:
                import paddle
                gpu_available = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
                if gpu_available:
                    logger.info("[PaddleOCR] CUDA GPU detected, PaddleOCR will use GPU")
                else:
                    logger.info("[PaddleOCR] No CUDA GPU available, using CPU")
            except Exception:
                logger.info("[PaddleOCR] Could not detect GPU, using CPU")
        
        # Determine MKL-DNN setting (for logging - paddle auto-detects)
        enable_mkldnn = settings.paddle_enable_mkldnn
        if enable_mkldnn is None:
            # Auto-detect: enable on x86 CPUs, disable on ARM (Apple Silicon)
            import platform
            enable_mkldnn = platform.machine() not in ('arm64', 'aarch64')
        
        logger.info("[PaddleOCR] Creating PaddleOCR instance (GPU available=%s, MKL-DNN=%s)...", gpu_available, enable_mkldnn)
        create_start = time.time()

        try:
            # PaddleOCR 3.x API - uses different parameter names
            _ocr_instance = PaddleOCR(
                use_doc_orientation_classify=False,  # Skip document orientation
                use_doc_unwarping=False,             # Skip document unwarping
                use_textline_orientation=False,      # Skip textline orientation
                lang=settings.paddle_ocr_lang,
                text_det_limit_side_len=736,         # Reduce detection image size (default 960)
                text_recognition_batch_size=6,       # Enable batch processing for recognition
            )
        except Exception as exc:
            _warmup_error = exc
            _warmup_done.set()
            raise

        create_time = time.time() - create_start
        total_time = time.time() - init_start
        _warmup_error = None
        _warmup_done.set()
        logger.info("[PaddleOCR] Instance created in %.2fs (total init: %.2fs)", create_time, total_time)
        return _ocr_instance


def ocr_pdf(path: str | Path) -> List[PageData]:
    """
    Process a PDF through PaddleOCR and return PageData.
    
    Args:
        path: Path to PDF file
    
    Returns:
        List of PageData objects with extracted text blocks
    """
    import time
    
    total_start = time.time()
    path = Path(path)
    logger.info("[PaddleOCR] Running OCR on PDF: %s", path)
    
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for OCR rendering. Install via `pip install PyMuPDF`."
        ) from exc
    
    # Get cached OCR instance (avoids slow model reloading)
    logger.debug("[PaddleOCR] Getting OCR instance...")
    ocr_start = time.time()
    ocr = _get_ocr()
    logger.debug("[PaddleOCR] Got OCR instance in %.2fs", time.time() - ocr_start)
    
    doc = fitz.open(path)
    total_pages = len(doc)
    
    # Use configurable DPI (default 100, reduced from 150 for speed)
    dpi = getattr(settings, 'paddle_render_dpi', 100)
    if dpi <= 0:
        dpi = 100
    
    logger.info("[PaddleOCR] Processing %d pages at %d DPI...", total_pages, dpi)
    
    def process_page(page_num: int) -> PageData:
        """Process a single page through OCR."""
        page_start = time.time()
        page = doc[page_num]
        page_num_display = page_num + 1
        
        logger.debug("[PaddleOCR] Page %d/%d: Rendering...", page_num_display, total_pages)
        
        # Render page at configured DPI (lower = faster)
        render_start = time.time()
        pix = page.get_pixmap(dpi=dpi)
        logger.debug("[PaddleOCR] Page %d: Rendered in %.2fs (%dx%d)", 
                    page_num_display, time.time() - render_start, pix.width, pix.height)
        
        # Convert pixmap to numpy array for PaddleOCR
        import numpy as np
        from PIL import Image
        
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_array = np.array(img)
        
        # Run OCR using predict() (new 3.x API)
        logger.debug("[PaddleOCR] Page %d: Running OCR...", page_num_display)
        ocr_start = time.time()
        ocr_result = ocr.predict(img_array)
        ocr_time = time.time() - ocr_start
        logger.debug("[PaddleOCR] Page %d: OCR took %.2fs", page_num_display, ocr_time)
        
        # Convert PaddleOCR results to TextBlocks
        text_blocks = _paddle_results_to_text_blocks(ocr_result, pix.width, pix.height, dpi=dpi)
        
        page_data = PageData(
            page_num=page_num_display,
            width=page.rect.width,
            height=page.rect.height,
            blocks=text_blocks,
        )
        page_data.metadata = {
            "extraction_method": "ocr_paddle",
            "ocr_engine_used": "paddle",
            "dpi": dpi,
        }
        
        page_time = time.time() - page_start
        logger.info("[PaddleOCR] Page %d/%d: %d blocks in %.2fs", 
                   page_num_display, total_pages, len(text_blocks), page_time)
        
        return page_data
    
    # Process pages sequentially (PaddlePaddle's predict() is not thread-safe)
    # The main performance gains come from:
    # - Reduced DPI (100 vs 150)
    # - Smaller detection image size (736 vs 960)
    # - Batch recognition
    pages: List[PageData] = []
    
    for i in range(total_pages):
        pages.append(process_page(i))
    
    doc.close()
    total_time = time.time() - total_start
    total_blocks = sum(len(p.blocks) for p in pages)
    logger.info("[PaddleOCR] Processed %d pages, %d blocks in %.2fs (%.2fs/page)", 
               len(pages), total_blocks, total_time, total_time / max(len(pages), 1))
    return pages


def _filter_contained_blocks(blocks: List[TextBlock]) -> List[TextBlock]:
    """
    Filter out blocks that strictly contain other blocks (e.g. table wrappers).
    Also filters duplicates.
    """
    if not blocks:
        return []
        
    n = len(blocks)
    to_remove = set()
    
    # Pre-calculate geometries (x, y, x2, y2, area)
    geoms = []
    for b in blocks:
        bx = b.bbox
        x, y = float(bx['x']), float(bx['y'])
        w, h = float(bx['width']), float(bx['height'])
        geoms.append((x, y, x + w, y + h, w * h))

    for i in range(n):
        if i in to_remove:
            continue
            
        xi1, yi1, xi2, yi2, area_i = geoms[i]
        
        for j in range(n):
            if i == j:
                continue
            if j in to_remove:
                continue
                
            xj1, yj1, xj2, yj2, area_j = geoms[j]
            
            # Intersection
            xx1 = max(xi1, xj1)
            yy1 = max(yi1, yj1)
            xx2 = min(xi2, xj2)
            yy2 = min(yi2, yj2)
            
            w_int = max(0.0, xx2 - xx1)
            h_int = max(0.0, yy2 - yy1)
            area_int = w_int * h_int
            
            if area_int <= 0:
                continue
                
            # Coverage relative to j (the smaller/contained one)
            coverage_j = area_int / area_j if area_j > 0 else 0.0
            
            # If j is effectively inside i (allow some margin for error)
            if coverage_j > 0.90:
                # If i is significantly larger (container) - e.g. table wrapper
                if area_i > 1.2 * area_j:
                    to_remove.add(i)
                    break 
                # If similar size (duplicate)
                # Remove j if i covers j and they are similar size
                else:
                    coverage_i = area_int / area_i if area_i > 0 else 0.0
                    if coverage_i > 0.90:
                         # Overlap is high for both -> Duplicate
                         # Remove the one with smaller area or arbitrary (j)
                         to_remove.add(j)
        
    return [b for k, b in enumerate(blocks) if k not in to_remove]


def _paddle_results_to_text_blocks(
    ocr_result: List,
    img_width: float,
    img_height: float,
    dpi: int = 100,
) -> List[TextBlock]:
    """
    Convert PaddleOCR 3.x results to TextBlock format with word-level metadata.
    
    PaddleOCR 3.x returns a list of dicts with keys:
    - 'rec_texts': list of recognized texts
    - 'rec_scores': list of confidence scores
    - 'dt_polys': list of polygon coordinates (4 points each)
    
    We convert to our format: {"x": x, "y": y, "width": w, "height": h}
    and include word-level bbox in metadata["words"].
    """
    text_blocks = []
    
    if not ocr_result:
        return text_blocks
    
    scale_factor = 72.0 / float(dpi)  # Convert DPI pixels to 72 DPI points
    
    # Group into lines with word metadata
    lines = _group_lines_from_results(ocr_result, scale_factor)
    
    for line_data in lines:
        text = line_data["text"]
        bbox = line_data["bbox"]
        words = line_data["words"]
        confidence = line_data["confidence"]
        
        block = TextBlock(
            text=text,
            bbox=bbox,
            style=None,
            metadata={
                "granularity": "line",
                "ocr_engine": "paddle",
                "bbox_source": "exact",
                "bbox_units": "pt",
                "bbox_space": "page",
                "confidence": confidence,
                "text_source": "paddle_ocr",
                "words": words,  # Word-level bbox metadata
            }
        )
        text_blocks.append(block)
    
    # Filter out excessive bboxes (e.g. table containers overlapping cells)
    return _filter_contained_blocks(text_blocks)
