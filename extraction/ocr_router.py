"""Multi-engine OCR router with hardware-aware engine selection and automatic fallback."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

from comparison.models import PageData
from config.settings import settings
from utils.logging import logger

# Track warmup state
_warmup_complete = threading.Event()
_warmup_thread: Optional[threading.Thread] = None


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    logger.debug("[OCR] Checking CUDA availability...")
    try:
        import torch
        result = torch.cuda.is_available()
        logger.debug("[OCR] CUDA available: %s", result)
        return result
    except ImportError:
        logger.debug("[OCR] CUDA check failed: torch not installed")
        return False


def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available (Mac M-series)."""
    logger.debug("[OCR] Checking MPS availability...")
    try:
        import torch
        result = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        logger.debug("[OCR] MPS available: %s", result)
        return result
    except ImportError:
        logger.debug("[OCR] MPS check failed: torch not installed")
        return False


def is_paddle_available() -> bool:
    """Check if PaddleOCR dependencies are available."""
    logger.debug("[OCR] Checking PaddleOCR availability...")
    try:
        import paddleocr
        logger.debug("[OCR] paddleocr module imported")
        # Also check for paddle (paddlepaddle)
        try:
            import paddle  # type: ignore
            logger.debug("[OCR] PaddleOCR available: True")
            return True
        except ImportError:
            logger.debug("[OCR] PaddleOCR: paddlepaddle not installed")
            return False
    except ImportError:
        logger.debug("[OCR] PaddleOCR: paddleocr not installed")
        return False


def is_tesseract_available() -> bool:
    """Check if Tesseract OCR is available."""
    logger.debug("[OCR] Checking Tesseract availability...")
    try:
        import pytesseract
        # Also check if tesseract binary is available
        try:
            version = pytesseract.get_tesseract_version()
            logger.debug("[OCR] Tesseract available: True (version: %s)", version)
            return True
        except Exception as e:
            logger.debug("[OCR] Tesseract binary not found: %s", e)
            return False
    except ImportError:
        logger.debug("[OCR] Tesseract: pytesseract not installed")
        return False


def is_deepseek_available() -> bool:
    """Check if DeepSeek-OCR is available."""
    logger.debug("[OCR] Checking DeepSeek availability...")
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        # Check if model path exists or can be loaded
        model_path = settings.deepseek_ocr_model_path
        if model_path and Path(model_path).exists():
            logger.debug("[OCR] DeepSeek available: True (local model at %s)", model_path)
            return True
        # Or check if it can be loaded from HuggingFace
        logger.debug("[OCR] DeepSeek available: True (will try HuggingFace)")
        return True  # Will fail at runtime if not available, but dependency check passes
    except ImportError:
        logger.debug("[OCR] DeepSeek: transformers not installed")
        return False


def warmup_ocr_engines(engines: Optional[List[str]] = None, background: bool = True) -> None:
    """
    Preload OCR engines to avoid slow first-request initialization.
    
    Args:
        engines: List of engines to warmup. If None, uses first available from priority.
        background: If True, run warmup in background thread.
    """
    global _warmup_thread
    
    logger.info("[OCR] Starting OCR warmup (background=%s)", background)
    
    if _warmup_complete.is_set():
        logger.debug("[OCR] Warmup already complete, skipping")
        return
    
    def _do_warmup():
        warmup_start = time.time()
        logger.info("[OCR] Warmup thread started")
        
        if engines is None:
            logger.debug("[OCR] No engines specified, detecting available engines...")
            available, skipped = select_ocr_engine(settings.ocr_engine_priority)
            target_engines = available[:1]  # Only warmup first available engine
            logger.info("[OCR] Available engines: %s, will warmup: %s", available, target_engines)
            if skipped:
                logger.info("[OCR] Skipped engines: %s", skipped)
        else:
            target_engines = engines
            logger.info("[OCR] Warming up specified engines: %s", target_engines)
        
        for engine_name in target_engines:
            engine_start = time.time()
            try:
                logger.info("[OCR] >>> Warming up engine: %s", engine_name)
                if engine_name == "paddle":
                    logger.debug("[OCR] Importing PaddleOCR...")
                    from extraction.paddle_ocr_engine import _get_ocr
                    logger.debug("[OCR] Calling _get_ocr() to initialize PaddleOCR...")
                    _get_ocr()  # Trigger lazy initialization
                elif engine_name == "tesseract":
                    # Tesseract uses native binary, just verify availability
                    logger.debug("[OCR] Verifying Tesseract binary...")
                    import pytesseract
                    pytesseract.get_tesseract_version()
                elif engine_name == "deepseek":
                    # Skip DeepSeek warmup - too heavy for background init
                    logger.info("[OCR] Skipping DeepSeek warmup (too heavy for background)")
                    continue
                engine_time = time.time() - engine_start
                logger.info("[OCR] <<< Engine %s warmed up in %.2fs", engine_name, engine_time)
            except Exception as e:
                engine_time = time.time() - engine_start
                logger.warning("[OCR] <<< Engine %s warmup FAILED after %.2fs: %s", engine_name, engine_time, e)
        
        _warmup_complete.set()
        total_time = time.time() - warmup_start
        logger.info("[OCR] Warmup complete in %.2fs", total_time)
    
    if background:
        logger.debug("[OCR] Starting warmup in background thread")
        _warmup_thread = threading.Thread(target=_do_warmup, daemon=True, name="OCRWarmup")
        _warmup_thread.start()
    else:
        _do_warmup()


def select_ocr_engine(
    engine_priority: List[str],
    hardware_available: Optional[dict] = None
) -> tuple[List[str], dict[str, str]]:
    """
    Select OCR engines based on hardware and dependency availability.
    
    Preflight check: skips engines if dependencies are not available.
    
    Args:
        engine_priority: List of engine names in priority order
        hardware_available: Optional dict with hardware info (auto-detected if None)
    
    Returns:
        Tuple of (filtered_engines, fallback_reasons) where fallback_reasons maps
        engine_name -> reason for skipping
    """
    logger.debug("[OCR] select_ocr_engine: priority=%s", engine_priority)
    
    if hardware_available is None:
        logger.debug("[OCR] Detecting hardware...")
        cuda_avail = is_cuda_available()
        mps_avail = is_mps_available()
        hardware_available = {"cuda": cuda_avail, "mps": mps_avail}
        logger.info("[OCR] Hardware: CUDA=%s, MPS=%s", cuda_avail, mps_avail)
    
    filtered_engines = []
    fallback_reasons = {}
    
    for engine in engine_priority:
        logger.debug("[OCR] Checking engine: %s", engine)
        skip_engine = False
        reason = None
        
        if engine == "deepseek":
            # Allow DeepSeek if either CUDA or MPS is available
            # Note: It can technically run on CPU, but it's very slow. 
            # We allow it on MPS now for Mac M-series users.
            has_cuda = hardware_available.get("cuda", False)
            has_mps = hardware_available.get("mps", False)
            
            if not (has_cuda or has_mps):
                # Optionally check settings if user forced CPU mode? 
                # For now, we still require some acceleration or explicitly allow CPU via setting?
                # The user "enabled" it, so we might want to relax this fully if they want CPU?
                # But typically DeepSeek is too heavy for pure CPU.
                # However, the user specifically asked for Mac M4 which has MPS.
                search_mps = is_mps_available() # Double check if not in dict
                if not search_mps:
                     # Relaxing: If no acceleration, warn but maybe allow if it's the ONLY option?
                     # Current logic: Skip if no acceleration.
                     skip_engine = True
                     reason = "cuda_or_mps_unavailable"
                     logger.debug("[OCR] DeepSeek: no CUDA/MPS acceleration")
            
            if not skip_engine and not is_deepseek_available():
                skip_engine = True
                reason = "dependency_missing"
        elif engine == "paddle":
            if not is_paddle_available():
                skip_engine = True
                reason = "dependency_missing"
        elif engine == "tesseract":
            if not is_tesseract_available():
                skip_engine = True
                reason = "dependency_missing"
        
        if skip_engine:
            logger.info("[OCR] Engine %s SKIPPED (reason: %s)", engine, reason)
            fallback_reasons[engine] = reason
            continue
        
        logger.info("[OCR] Engine %s AVAILABLE", engine)
        filtered_engines.append(engine)
    
    logger.debug("[OCR] select_ocr_engine result: available=%s, skipped=%s", 
                filtered_engines, fallback_reasons)
    return filtered_engines, fallback_reasons


def ocr_pdf_multi(
    path: str | Path,
    engine_priority: Optional[List[str]] = None,
) -> List[PageData]:
    """
    Scanned PDF OCR su multi-engine fallback.
    
    Garantija: visada grąžina PageData su užpildyta metadata apie engine ir fallback.
    
    Args:
        path: PDF file path
        engine_priority: Custom priority list ["deepseek", "paddle", "tesseract"]
                        If None, uses settings.ocr_engine_priority
    
    Returns:
        List[PageData] su metadata:
        - extraction_method: "ocr_deepseek" | "ocr_paddle" | "ocr_tesseract"
        - ocr_engine_used: engine name
        - ocr_fallback_reason: reason if fallback occurred (optional)
        - dpi: used DPI
    """
    total_start = time.time()
    path = Path(path)
    logger.info("[OCR] ========== Starting multi-engine OCR ==========")
    logger.info("[OCR] PDF: %s", path)
    logger.info("[OCR] Warmup complete: %s", _warmup_complete.is_set())
    
    # Get engine priority from settings if not provided
    if engine_priority is None:
        engine_priority = settings.ocr_engine_priority
    logger.info("[OCR] Engine priority: %s", engine_priority)
    
    # Filter engines based on hardware and dependencies (preflight check)
    logger.debug("[OCR] Running engine preflight checks...")
    preflight_start = time.time()
    available_engines, fallback_reasons = select_ocr_engine(engine_priority)
    preflight_time = time.time() - preflight_start
    logger.info("[OCR] Preflight completed in %.2fs", preflight_time)
    logger.info("[OCR] Available engines: %s", available_engines)
    
    if not available_engines:
        logger.error(
            "[OCR] No OCR engines available! Skipped: %s",
            fallback_reasons
        )
        return []
    
    # Log which engines were skipped and why
    if fallback_reasons:
        logger.warning("[OCR] Skipped engines: %s", fallback_reasons)
    
    # Try each engine in priority order
    last_error = None
    attempted_engines = []
    
    for idx, engine_name in enumerate(available_engines):
        engine_start = time.time()
        logger.info("[OCR] >>> Attempting engine %d/%d: %s", idx + 1, len(available_engines), engine_name)
        attempted_engines.append(engine_name)
        
        try:
            pages = _run_ocr_engine(path, engine_name)
            engine_time = time.time() - engine_start
            
            # Check if we got sufficient results
            total_blocks = sum(len(p.blocks) for p in pages)
            total_chars = sum(len(b.text) for p in pages for b in p.blocks)
            logger.info("[OCR] Engine %s returned %d pages, %d blocks, %d chars in %.2fs",
                       engine_name, len(pages), total_blocks, total_chars, engine_time)
            
            if _is_ocr_successful(pages):
                total_time = time.time() - total_start
                logger.info("[OCR] <<< SUCCESS with engine: %s (total time: %.2fs)", engine_name, total_time)
                # If this wasn't the first engine, mark fallback
                if idx > 0:
                    previous_engine = attempted_engines[-2]
                    fallback_reason = fallback_reasons.get(previous_engine, "unknown")
                    _mark_fallback_reason(pages, attempted_engines, fallback_reason)
                logger.info("[OCR] ========== OCR complete ==========")
                return pages
            
            # Check if we should fallback
            logger.warning(
                "[OCR] <<< Engine %s produced insufficient results (%d blocks, %d chars). Trying next...",
                engine_name, total_blocks, total_chars
            )
            last_error = "insufficient_blocks"
            
        except Exception as e:
            engine_time = time.time() - engine_start
            logger.error("[OCR] <<< Engine %s FAILED after %.2fs: %s", engine_name, engine_time, e)
            last_error = str(e)
            continue
    
    # All engines failed or produced insufficient results
    total_time = time.time() - total_start
    logger.error("[OCR] All OCR engines failed after %.2fs", total_time)
    logger.info("[OCR] ========== OCR failed ==========")
    
    # Return empty pages with metadata about failure
    return _create_empty_pages_with_metadata(path, available_engines, last_error)


def _run_ocr_engine(path: Path, engine_name: str) -> List[PageData]:
    """Run OCR with a specific engine."""
    logger.debug("[OCR] _run_ocr_engine: importing %s engine...", engine_name)
    import_start = time.time()
    
    if engine_name == "deepseek":
        from extraction.deepseek_ocr_engine import ocr_pdf as deepseek_ocr_pdf
        logger.debug("[OCR] DeepSeek import took %.2fs", time.time() - import_start)
        logger.debug("[OCR] Running DeepSeek OCR...")
        return deepseek_ocr_pdf(path, settings.deepseek_ocr_model_path)
    elif engine_name == "paddle":
        from extraction.paddle_ocr_engine import ocr_pdf as paddle_ocr_pdf
        logger.debug("[OCR] PaddleOCR import took %.2fs", time.time() - import_start)
        logger.debug("[OCR] Running PaddleOCR...")
        return paddle_ocr_pdf(path)
    elif engine_name == "tesseract":
        from extraction.tesseract_ocr_engine import ocr_pdf as tesseract_ocr_pdf
        logger.debug("[OCR] Tesseract import took %.2fs", time.time() - import_start)
        logger.debug("[OCR] Running Tesseract OCR...")
        return tesseract_ocr_pdf(path)
    else:
        raise ValueError(f"Unknown OCR engine: {engine_name}")


def _is_ocr_successful(pages: List[PageData]) -> bool:
    """Check if OCR produced sufficient results."""
    min_blocks = settings.min_ocr_blocks_per_page
    
    for page in pages:
        if len(page.blocks) >= min_blocks:
            return True
    
    # Check total character count as fallback
    total_chars = sum(len(block.text) for page in pages for block in page.blocks)
    if total_chars >= 30:  # At least 30 characters total
        return True
    
    return False


def _mark_fallback_reason(
    pages: List[PageData], 
    attempted_engines: List[str],
    fallback_reason: Optional[str] = None
) -> None:
    """Mark pages with fallback reason when engine switch occurred."""
    if len(attempted_engines) < 2:
        return
    
    previous_engine = attempted_engines[-2]
    current_engine = attempted_engines[-1]
    
    # Standardized fallback reason
    reason = fallback_reason or f"switched_from_{previous_engine}"
    
    for page in pages:
        if "ocr_fallback_reason" not in page.metadata:
            page.metadata["ocr_fallback_reason"] = reason


def _create_empty_pages_with_metadata(
    path: Path,
    attempted_engines: List[str],
    last_error: Optional[str]
) -> List[PageData]:
    """Create empty PageData objects with failure metadata."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF required for page metadata")
        return []
    
    doc = fitz.open(path)
    pages: List[PageData] = []
    
    for page in doc:
        page_data = PageData(
            page_num=page.number + 1,
            width=page.rect.width,
            height=page.rect.height,
            blocks=[],
        )
        page_data.metadata = {
            "extraction_method": f"ocr_{attempted_engines[-1] if attempted_engines else 'unknown'}",
            "ocr_engine_used": attempted_engines[-1] if attempted_engines else "none",
            "ocr_fallback_reason": last_error or "all_engines_failed",
            "dpi": 300,
        }
        pages.append(page_data)
    
    doc.close()
    return pages
