"""Multi-engine OCR router with hardware-aware engine selection and automatic fallback.

This module provides the main entry point for OCR processing with:
- Hardware detection (CUDA, MPS, CPU)
- Multi-engine fallback (DeepSeek → Tesseract → PaddleOCR)
- DeepSeek guardrail integration (timeout, memory limits, academic-safe validation)
- Digital PDF fast path (skip OCR for born-digital PDFs)
"""
from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from comparison.models import PageData
from config.settings import settings
from utils.coordinates import normalize_page_bboxes
from utils.logging import logger

# Track warmup state
_warmup_complete = threading.Event()
_warmup_thread: Optional[threading.Thread] = None

# Layout analysis setting
_run_layout_after_ocr = True  # Enable layout analysis for OCR output


def _normalize_engine_name(name: str) -> str:
    return str(name or "").strip().lower()


def _classify_failure_reason(err: object | None) -> str:
    """Map exceptions / error strings to stable failure_reason tokens."""
    if err is None:
        return "unknown"
    if isinstance(err, TimeoutError):
        return "engine_timeout"
    msg = str(err).lower()
    if "timeout" in msg:
        return "engine_timeout"
    if "memory" in msg or "rss" in msg or "oom" in msg:
        return "engine_memory"
    if "insufficient" in msg or "min_ocr_blocks" in msg:
        return "insufficient_blocks"
    return "engine_error"


def _apply_scanned_policy_to_priority(
    engine_priority: List[str] | None,
    *,
    policy: str,
) -> tuple[List[str], bool]:
    """Return (effective_priority, allow_fallback)."""
    policy_norm = _normalize_engine_name(policy) or "strict"

    # If caller didn't pass a list, default to AUTO chain (auto_fallback) or selected engine.
    if engine_priority is None:
        if policy_norm == "auto_fallback":
            # NOTE:
            # For scanned PDFs, deterministic engine order matters a lot for diff quality.
            # In auto_fallback, prefer the configured scanned chain as-is; users can
            # override order by editing `ocr_scanned_default_chain` or by passing an
            # explicit `engine_priority`.
            chain = getattr(settings, "ocr_scanned_default_chain", None)
            if isinstance(chain, list) and chain:
                engine_priority = list(chain)
            else:
                engine_priority = ["tesseract", "paddle", "deepseek"]
        else:
            engine_priority = [settings.ocr_engine] if getattr(settings, "ocr_engine", None) else []
    else:
        engine_priority = list(engine_priority)

    engine_priority = [_normalize_engine_name(e) for e in engine_priority if _normalize_engine_name(e)]
    if not engine_priority:
        engine_priority = ["paddle"]

    if policy_norm == "strict":
        return engine_priority[:1], False

    # auto_fallback
    allow_fallback = True

    # If DeepSeek is explicitly first, ensure we have a deterministic fallback chain.
    if engine_priority and engine_priority[0] == "deepseek":
        chain = getattr(settings, "ocr_scanned_fallback_chain", None)
        if not (isinstance(chain, list) and chain):
            chain = ["tesseract", "paddle"]
        expanded = ["deepseek"]
        for e in chain:
            en = _normalize_engine_name(e)
            if en and en != "deepseek" and en not in expanded:
                expanded.append(en)
        # Preserve any extra engines the caller provided (after deterministic chain)
        for e in engine_priority[1:]:
            if e not in expanded:
                expanded.append(e)
        engine_priority = expanded

    return engine_priority, allow_fallback


def _filter_contained_blocks(blocks: List[Any]) -> List[Any]:
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
        # b can be TextBlock object or dict
        if isinstance(b, dict):
             bx = b.get("bbox", {})
        else:
             bx = getattr(b, "bbox", {})
             
        x, y = float(bx.get('x', 0)), float(bx.get('y', 0))
        w, h = float(bx.get('width', 0)), float(bx.get('height', 0))
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
            
            # If j is effectively inside i
            if coverage_j > 0.90:
                # If i is significantly larger (container) - e.g. table wrapper
                if area_i > 1.2 * area_j:
                    to_remove.add(i)
                    break 
                # If similar size (duplicate)
                else:
                    coverage_i = area_int / area_i if area_i > 0 else 0.0
                    if coverage_i > 0.90:
                         to_remove.add(j)
        
    return [b for k, b in enumerate(blocks) if k not in to_remove]


def _annotate_routing_metadata(
    pages: List[PageData],
    *,
    policy: str,
    engine_selected: str,
    attempts: List[Dict[str, Any]],
    engine_priority: Optional[List[str]] = None,
    available_engines: Optional[List[str]] = None,
    preflight_skipped: Optional[Dict[str, str]] = None,
    status: str,
    failure_reason: Optional[str],
) -> None:
    for page in pages:
        md = page.metadata or {}
        md.setdefault("ocr_policy", policy)
        md.setdefault("ocr_engine_selected", engine_selected)
        md.setdefault("ocr_attempts", attempts)
        if engine_priority is not None:
            md.setdefault("ocr_engine_priority", list(engine_priority))
        if available_engines is not None:
            md.setdefault("ocr_available_engines", list(available_engines))
        if preflight_skipped is not None:
            md.setdefault("ocr_preflight_skipped", dict(preflight_skipped))
        md.setdefault("ocr_status", status)
        if failure_reason:
            md.setdefault("ocr_failure_reason", failure_reason)
        page.metadata = md


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
    if importlib.util.find_spec("paddleocr") is None:
        logger.debug("[OCR] PaddleOCR: paddleocr not installed")
        return False
    if importlib.util.find_spec("paddle") is None:
        logger.debug("[OCR] PaddleOCR: paddlepaddle not installed")
        return False
    logger.debug("[OCR] PaddleOCR available: True")
    return True


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
    if importlib.util.find_spec("transformers") is None:
        logger.debug(
            "[OCR] DeepSeek: transformers not installed. Ensure you are running in the virtual environment."
        )
        return False

    # Check if model path exists or can be loaded
    model_path = settings.deepseek_ocr_model_path
    if model_path and Path(model_path).exists():
        logger.debug("[OCR] DeepSeek available: True (local model at %s)", model_path)
        return True

    # Or check if it can be loaded from HuggingFace
    logger.debug("[OCR] DeepSeek available: True (will try HuggingFace)")
    return True  # Will fail at runtime if not available, but dependency check passes


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
            logger.debug("[OCR] No engines specified, using configured engine...")
            target_engine = settings.ocr_engine
            available, skipped = select_ocr_engine([target_engine])
            target_engines = available[:1]  # Should be just the one if available
            logger.info("[OCR] Configured engine: %s, available: %s", target_engine, available)
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
    run_layout_analysis: bool = True,
    prefer_digital: bool = True,
) -> List[PageData]:
    """
    Scanned PDF OCR su multi-engine fallback.
    
    Garantija: visada grąžina PageData su užpildyta metadata apie engine ir fallback.
    
    Args:
        path: PDF file path
        engine_priority: Custom priority list ["deepseek", "paddle", "tesseract"]
                        If None, uses settings.ocr_engine_priority
        run_layout_analysis: Whether to run layout analysis after OCR
    
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
    logger.info("[OCR] PDF: %s (layout_analysis=%s)", path, run_layout_analysis)
    logger.info("[OCR] Warmup complete: %s", _warmup_complete.is_set())
    
    # ---------------------------------------------------------------------
    # DIGITAL-FIRST FAST PATH
    # If PDF has a real text layer, use PyMuPDF extraction instead of OCR.
    # This massively reduces noise and preserves Lithuanian chars + fonts.
    # ---------------------------------------------------------------------
    if prefer_digital:
        digital_pages = _try_digital_extract(path, run_layout_analysis=run_layout_analysis)
        if digital_pages:
            total_time = time.time() - total_start
            logger.info("[OCR] PDF looks digital -> using PyMuPDF text extraction (no OCR) in %.2fs.", total_time)
            return digital_pages
    
    # Get engine priority from settings if not provided
    caller_provided_priority = engine_priority is not None
    if engine_priority is None:
        # Legacy defaults:
        # - if scanned policy is strict: use selected engine
        # - if auto_fallback: use ocr_scanned_default_chain
        pass

    # Apply scanned routing policy (strict vs auto_fallback)
    policy = getattr(settings, "ocr_scanned_policy", "strict")
    engine_priority, allow_fallback = _apply_scanned_policy_to_priority(engine_priority, policy=policy)

    # Backward-compat: honor old ocr_fallback_enabled by forcing strict behavior
    if not getattr(settings, "ocr_fallback_enabled", True):
        engine_priority = engine_priority[:1]
        allow_fallback = False
        logger.info("[OCR] Fallback disabled (legacy setting). Restricted to: %s", engine_priority)

    # Only when caller did NOT specify a priority: prepend the selected engine (legacy behavior)
    # for non-AUTO usage.
    if (not caller_provided_priority) and getattr(settings, "ocr_engine", None) and engine_priority:
        selected = _normalize_engine_name(settings.ocr_engine)
        if selected and selected != engine_priority[0] and policy != "auto_fallback":
            if selected in engine_priority:
                engine_priority.remove(selected)
            engine_priority.insert(0, selected)

    logger.info("[OCR] Engine priority: %s", engine_priority)
    if not allow_fallback:
        logger.info("[OCR] Routing policy: strict (no fallback)")
    else:
        logger.info("[OCR] Routing policy: auto_fallback")
    
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
    attempts: List[Dict[str, Any]] = []
    engine_selected = engine_priority[0] if engine_priority else ""
    
    for idx, engine_name in enumerate(available_engines):
        engine_start = time.time()
        logger.info("[OCR] >>> Attempting engine %d/%d: %s", idx + 1, len(available_engines), engine_name)
        attempted_engines.append(engine_name)
        
        try:
            pages = _run_ocr_engine(path, engine_name)
            pages = normalize_page_bboxes(pages)  # bbox police checkpoint ✅

            # Filter contained blocks (deduplication/cleanup)
            for page in pages:
                page.blocks = _filter_contained_blocks(page.blocks)

            # Annotate OCR quality for downstream gating/weighting
            _annotate_ocr_quality(pages, engine_name=engine_name)
            engine_time = time.time() - engine_start
            
            # Check if we got sufficient results
            total_blocks = sum(len(p.blocks) for p in pages)
            total_chars = sum(len(b.text) for p in pages for b in p.blocks)
            logger.info("[OCR] Engine %s returned %d pages, %d blocks, %d chars in %.2fs",
                       engine_name, len(pages), total_blocks, total_chars, engine_time)
            
            if _is_ocr_successful(pages, engine_name):
                total_time = time.time() - total_start
                logger.info("[OCR] <<< SUCCESS with engine: %s (total time: %.2fs)", engine_name, total_time)

                attempts.append({
                    "engine": engine_name,
                    "outcome": "success",
                    "elapsed_sec": round(engine_time, 3),
                })

                # If this wasn't the first engine, mark fallback
                if idx > 0:
                    previous_engine = attempted_engines[-2]
                    fallback_reason = fallback_reasons.get(previous_engine, "unknown")
                    _mark_fallback_reason(pages, attempted_engines, fallback_reason)
                
                # Run layout analysis to populate tables/figures metadata
                if run_layout_analysis and _run_layout_after_ocr:
                    pages = _add_layout_metadata(path, pages)

                _annotate_routing_metadata(
                    pages,
                    policy=str(policy),
                    engine_selected=str(engine_selected),
                    attempts=list(attempts),
                    engine_priority=list(engine_priority) if engine_priority else [],
                    available_engines=list(available_engines) if available_engines else [],
                    preflight_skipped=dict(fallback_reasons) if fallback_reasons else {},
                    status="ok",
                    failure_reason=None,
                )
                
                logger.info("[OCR] ========== OCR complete ==========")
                return pages
            
            # Check if we should fallback
            attempts.append({
                "engine": engine_name,
                "outcome": "insufficient",
                "elapsed_sec": round(engine_time, 3),
                "reason": "insufficient_blocks",
                "blocks": int(total_blocks),
                "chars": int(total_chars),
            })
            last_error = "insufficient_blocks"

            if not allow_fallback:
                logger.error(
                    "[OCR] Strict mode: engine %s produced insufficient results (%d blocks, %d chars). No fallback.",
                    engine_name, total_blocks, total_chars,
                )
                break

            logger.warning(
                "[OCR] <<< Engine %s produced insufficient results (%d blocks, %d chars). Trying next...",
                engine_name, total_blocks, total_chars,
            )
            
        except Exception as e:
            engine_time = time.time() - engine_start
            logger.error("[OCR] <<< Engine %s FAILED after %.2fs: %s", engine_name, engine_time, e)
            attempts.append({
                "engine": engine_name,
                "outcome": "failed",
                "elapsed_sec": round(engine_time, 3),
                "reason": _classify_failure_reason(e),
                "error": str(e),
                "error_type": type(e).__name__,
            })
            last_error = str(e)

            if not allow_fallback:
                logger.error("[OCR] Strict mode: %s failed (%s). No fallback.", engine_name, type(e).__name__)
                break
            continue
    
    # All engines failed or produced insufficient results
    total_time = time.time() - total_start
    logger.error("[OCR] All OCR engines failed after %.2fs", total_time)
    logger.info("[OCR] ========== OCR failed ==========")
    
    # Return empty pages with metadata about failure
    pages = _create_empty_pages_with_metadata(
        path,
        available_engines,
        last_error,
        policy=str(policy),
        engine_selected=str(engine_selected),
        attempts=list(attempts),
        engine_priority=list(engine_priority) if engine_priority else [],
        available_engines=list(available_engines) if available_engines else [],
        preflight_skipped=dict(fallback_reasons) if fallback_reasons else {},
        failure_reason=_classify_failure_reason(last_error),
    )
    return pages


def _run_deepseek_with_guardrails(path: Path) -> List[PageData]:
    """
    Run DeepSeek OCR with guardrails (timeout, memory limits, academic-safe validation).
    
    Processes pages one at a time with:
    - Per-page timeout enforcement (subprocess-based hard kill)
    - Memory monitoring
    - Two-tier academic-safe validation
    - Page budget enforcement (max_pages_per_doc)
    
    On GuardrailViolation, raises to trigger fallback to next engine.
    """
    from extraction.deepseek_ocr_engine import (
        GuardrailViolation,
        GuardrailResult,
        get_ocr_instance,
    )
    
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PyMuPDF required for PDF rendering") from exc
    
    logger.info("[OCR] Running DeepSeek with guardrails...")
    
    # Get guardrail settings
    max_pages = getattr(settings, "deepseek_max_pages_per_doc", 3)
    render_dpi = int(getattr(settings, "deepseek_render_dpi", 60))
    
    doc = fitz.open(path)
    total_pages = doc.page_count
    
    # Page budgeting: limit number of pages processed
    pages_to_process = min(total_pages, max_pages)
    if pages_to_process < total_pages:
        logger.warning(
            "[OCR] DeepSeek page budget: processing %d/%d pages (max_pages_per_doc=%d)",
            pages_to_process, total_pages, max_pages
        )
    
    ocr = get_ocr_instance(settings.deepseek_ocr_model_path)
    pages: List[PageData] = []
    failed_pages: List[int] = []
    guardrail_warnings: List[str] = []
    
    for page_idx in range(total_pages):
        page = doc.load_page(page_idx)
        page_num = page_idx + 1
        target_size = (page.rect.width, page.rect.height)
        
        # Create manifest dict to track engine_type
        manifest_page: Dict[str, Any] = {"page_num": page_num}
        
        if page_idx >= pages_to_process:
            # Over budget - create empty page with metadata
            logger.info("[OCR] Page %d: skipped (over budget)", page_num)
            page_data = PageData(
                page_num=page_num,
                width=page.rect.width,
                height=page.rect.height,
                blocks=[],
            )
            page_data.metadata = {
                "extraction_method": "ocr_deepseek",
                "ocr_engine_used": "deepseek",
                "engine_type": "deepseek",
                "deepseek_skipped": True,
                "deepseek_skip_reason": "page_budget_exceeded",
                "dpi": render_dpi,
            }
            pages.append(page_data)
            continue
        
        try:
            # Render page to image
            pix = page.get_pixmap(dpi=render_dpi, alpha=False)
            
            # Run guardrail-enabled OCR
            result: GuardrailResult = ocr.recognize_page(
                image=pix,
                page_index=page_idx,
                manifest_page=manifest_page,
                target_size=target_size,
            )
            
            if result.ok:
                # Success - create PageData
                page_data = PageData(
                    page_num=page_num,
                    width=page.rect.width,
                    height=page.rect.height,
                    blocks=result.blocks,
                )
                page_data.metadata = {
                    "extraction_method": "ocr_deepseek",
                    "ocr_engine_used": "deepseek",
                    "engine_type": manifest_page.get("engine_type", "deepseek"),
                    "deepseek_mode": manifest_page.get("deepseek_mode", "unknown"),
                    "deepseek_elapsed_sec": result.engine_meta.get("elapsed_sec", 0),
                    "deepseek_peak_rss_mb": result.engine_meta.get("peak_rss_mb", 0),
                    "dpi": render_dpi,
                }
                if result.warnings:
                    page_data.metadata["deepseek_warnings"] = result.warnings
                    guardrail_warnings.extend(result.warnings)
                
                pages.append(page_data)
                logger.info(
                    "[OCR] Page %d: DeepSeek OK (%d blocks, %.1fs)",
                    page_num, len(result.blocks), result.engine_meta.get("elapsed_sec", 0)
                )
            else:
                # Guardrail rejection (not exception) - mark as failed
                failed_pages.append(page_num)
                diag_reason = result.diagnostics.reason if result.diagnostics else "unknown"
                
                page_data = PageData(
                    page_num=page_num,
                    width=page.rect.width,
                    height=page.rect.height,
                    blocks=[],
                )
                page_data.metadata = {
                    "extraction_method": "ocr_deepseek",
                    "ocr_engine_used": "deepseek",
                    "engine_type": "deepseek",
                    "deepseek_failed": True,
                    "deepseek_fail_reason": diag_reason,
                    "dpi": render_dpi,
                }
                pages.append(page_data)
                logger.warning("[OCR] Page %d: DeepSeek rejected (%s)", page_num, diag_reason)
                
        except GuardrailViolation as gv:
            # Hard failure - log and re-raise to trigger engine fallback
            logger.error(
                "[OCR] Page %d: GuardrailViolation (%s). Triggering engine fallback.",
                page_num, gv.reason
            )
            doc.close()
            raise  # Propagate to trigger fallback to tesseract/paddle
            
        except Exception as e:
            # Unexpected error - mark page as failed but continue
            failed_pages.append(page_num)
            logger.error("[OCR] Page %d: unexpected error: %s", page_num, e)
            
            page_data = PageData(
                page_num=page_num,
                width=page.rect.width,
                height=page.rect.height,
                blocks=[],
            )
            page_data.metadata = {
                "extraction_method": "ocr_deepseek",
                "ocr_engine_used": "deepseek",
                "engine_type": "deepseek",
                "deepseek_failed": True,
                "deepseek_fail_reason": f"exception:{type(e).__name__}",
                "dpi": render_dpi,
            }
            pages.append(page_data)
    
    doc.close()
    
    # Log summary
    success_pages = len(pages) - len(failed_pages) - (total_pages - pages_to_process)
    logger.info(
        "[OCR] DeepSeek guardrails complete: %d/%d pages OK, %d failed, %d skipped",
        success_pages, total_pages, len(failed_pages), total_pages - pages_to_process
    )
    
    if guardrail_warnings:
        logger.info("[OCR] Guardrail warnings: %s", guardrail_warnings[:5])
    
    # If too many pages failed, consider raising to trigger fallback
    failure_ratio = len(failed_pages) / pages_to_process if pages_to_process > 0 else 0
    max_failure_ratio = getattr(settings, "deepseek_max_failure_ratio", 0.5)
    
    if failure_ratio > max_failure_ratio:
        logger.warning(
            "[OCR] DeepSeek failure ratio %.1f%% > %.1f%% threshold. May want to fallback.",
            failure_ratio * 100, max_failure_ratio * 100
        )
    
    return pages


def _run_ocr_engine(path: Path, engine_name: str) -> List[PageData]:
    """Run OCR with a specific engine.
    
    For DeepSeek, uses guardrail-enabled page-by-page processing with
    automatic fallback on GuardrailViolation.
    """
    logger.debug("[OCR] _run_ocr_engine: importing %s engine...", engine_name)
    import_start = time.time()
    
    if engine_name == "deepseek":
        # Check if guardrails are enabled
        use_guardrails = getattr(settings, "deepseek_enabled", True)
        
        if use_guardrails:
            return _run_deepseek_with_guardrails(path)
        else:
            # Legacy mode without guardrails
            from extraction.deepseek_ocr_engine import ocr_pdf as deepseek_ocr_pdf
            logger.debug("[OCR] DeepSeek import took %.2fs", time.time() - import_start)
            logger.debug("[OCR] Running DeepSeek OCR (legacy mode)...")
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


def _is_ocr_successful(pages: List[PageData], engine_name: str = "unknown") -> bool:
    """Check if OCR produced sufficient results.
    
    Args:
        pages: List of PageData from OCR processing
        engine_name: Name of the OCR engine used (affects thresholds)
    """
    # DeepSeek produces fewer but richer blocks, so use different thresholds
    if engine_name == "deepseek":
        min_blocks = max(1, settings.min_ocr_blocks_per_page // 2)
        min_chars = 100  # Higher char requirement for DeepSeek
    else:
        min_blocks = settings.min_ocr_blocks_per_page
        min_chars = 30  # Standard threshold for Paddle/Tesseract
    
    for page in pages:
        if len(page.blocks) >= min_blocks:
            # If we have quality metrics, reject pages that are entirely low-confidence.
            # This keeps OCR noise from becoming downstream false diffs.
            if not page.metadata.get("ocr_quality", {}).get("low_confidence", False):
                return True
    
    # Check total character count as fallback
    total_chars = sum(len(block.text) for page in pages for block in page.blocks)
    if total_chars >= min_chars:
        # If *all* pages are flagged low-quality, treat OCR as unsuccessful.
        # (This helps trigger engine fallback.)
        if pages and all(p.metadata.get("ocr_quality", {}).get("low_confidence", False) for p in pages):
            return False
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
    # current engine is attempted_engines[-1]
    
    # Standardized fallback reason
    reason = fallback_reason or f"switched_from_{previous_engine}"
    
    for page in pages:
        if "ocr_fallback_reason" not in page.metadata:
            page.metadata["ocr_fallback_reason"] = reason


def _create_empty_pages_with_metadata(
    path: Path,
    attempted_engines: List[str],
    last_error: Optional[str],
    *,
    policy: str,
    engine_selected: str,
    attempts: List[Dict[str, Any]],
    engine_priority: Optional[List[str]] = None,
    available_engines: Optional[List[str]] = None,
    preflight_skipped: Optional[Dict[str, str]] = None,
    failure_reason: str,
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
            "ocr_policy": policy,
            "ocr_engine_selected": engine_selected,
            "ocr_attempts": attempts,
            "ocr_engine_priority": list(engine_priority) if engine_priority is not None else [],
            "ocr_available_engines": list(available_engines) if available_engines is not None else list(attempted_engines),
            "ocr_preflight_skipped": dict(preflight_skipped) if preflight_skipped is not None else {},
            "ocr_status": "failed",
            "ocr_failure_reason": failure_reason,
            "dpi": _get_engine_render_dpi(attempted_engines[-1]) if attempted_engines else _get_engine_render_dpi(settings.ocr_engine),
        }
        pages.append(page_data)
    
    doc.close()
    return pages


def _add_layout_metadata(path: Path, pages: List[PageData]) -> List[PageData]:
    """
    Run layout analysis and merge tables/figures metadata into OCR pages.
    
    This ensures that table_comparison and figure_comparison can work
    with OCR-extracted pages just like digital PDF pages.
    """
    try:
        from extraction.layout_analyzer import analyze_layout
        
        layout_start = time.time()
        logger.info("[OCR] Running layout analysis for OCR output...")
        
        layout_pages = analyze_layout(path)
        layout_lookup = {p.page_num: p for p in layout_pages}
        
        # Merge layout metadata into OCR pages
        for page in pages:
            if page.page_num in layout_lookup:
                layout_page = layout_lookup[page.page_num]
                page.metadata.update({
                    "tables": layout_page.metadata.get("tables", []),
                    "figures": layout_page.metadata.get("figures", []),
                    "text_blocks": layout_page.metadata.get("text_blocks", []),
                    "layout_analyzed": True,
                    "layout_method": layout_page.metadata.get("layout_method", "yolo"),
                })
            else:
                # Mark as analyzed even if not found (edge case)
                page.metadata["layout_analyzed"] = True
                page.metadata["tables"] = []
                page.metadata["figures"] = []
        
        layout_time = time.time() - layout_start
        logger.info("[OCR] Layout analysis completed in %.2fs", layout_time)
        
    except Exception as e:
        logger.warning("[OCR] Layout analysis failed: %s. Continuing without.", e)
        # Mark pages as having no layout (prevents retrying)
        for page in pages:
            if "layout_analyzed" not in page.metadata:
                page.metadata["layout_analyzed"] = False
                page.metadata["layout_error"] = str(e)
                page.metadata["tables"] = []
                page.metadata["figures"] = []
    
    return pages


def _get_engine_render_dpi(engine_name: Optional[str]) -> int:
    """Return the configured render DPI for a given OCR engine."""
    name = (engine_name or "").lower()
    if name == "deepseek":
        dpi = int(getattr(settings, "deepseek_render_dpi", 60))
    elif name == "tesseract":
        dpi = int(getattr(settings, "tesseract_render_dpi", 150))
    else:
        # Default to Paddle settings for unknown/other engines
        dpi = int(getattr(settings, "paddle_render_dpi", 150))
    return dpi if dpi > 0 else 150


def _annotate_ocr_quality(pages: List[PageData], *, engine_name: str) -> None:
    """Attach per-page OCR quality metrics and a low-confidence flag."""
    min_chars = int(getattr(settings, "ocr_quality_min_chars_per_page", 25))
    min_conf = float(getattr(settings, "ocr_quality_min_avg_confidence", 0.55))
    max_gib = float(getattr(settings, "ocr_quality_max_gibberish_ratio", 0.35))

    for page in pages:
        text = "\n".join(b.text for b in page.blocks if b.text)
        char_count = len(text)

        conf_values: List[float] = []
        for block in page.blocks:
            c = block.metadata.get("confidence")
            if c is None:
                continue
            try:
                cf = float(c)
            except Exception:
                continue
            if cf > 1.0:
                cf = cf / 100.0
            if 0.0 <= cf <= 1.0:
                conf_values.append(cf)

        avg_conf = sum(conf_values) / len(conf_values) if conf_values else None

        if char_count:
            non_printable = sum(1 for ch in text if not ch.isprintable())
            replacement = text.count("\ufffd") + text.count("�")
            # Count characters that are neither alnum, whitespace, nor common punctuation.
            common_punct = set(".,;:!?()[]{}<>/\\-–—'\"%$€@#&*_+=|")
            weird = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in common_punct))
            gibberish_ratio = (non_printable + replacement + weird) / char_count
        else:
            gibberish_ratio = 1.0

        low_conf = False
        if char_count < min_chars:
            low_conf = True
        if avg_conf is not None and avg_conf < min_conf:
            low_conf = True
        if gibberish_ratio > max_gib:
            low_conf = True

        page.metadata["ocr_quality"] = {
            "engine": engine_name,
            "char_count": char_count,
            "avg_confidence": avg_conf,
            "gibberish_ratio": gibberish_ratio,
            "low_confidence": low_conf,
        }
        # Convenience top-level flag for quick checks
        page.metadata["ocr_low_confidence"] = low_conf


def _try_digital_extract(path: Path, *, run_layout_analysis: bool) -> List[PageData] | None:
    """
    Heuristic: if the PDF has enough selectable text, skip OCR and parse with PyMuPDF.
    
    Returns extracted pages if PDF is born-digital, None otherwise (triggers OCR).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.debug("[OCR] PyMuPDF not available for digital extraction check")
        return None

    try:
        with fitz.open(path) as doc:
            if doc.page_count == 0:
                return None
            # Sample first 2 pages (cheap + robust)
            sample_n = min(2, doc.page_count)
            chars = 0
            for i in range(sample_n):
                txt = doc.load_page(i).get_text("text") or ""
                chars += len(txt.strip())
            # Tuneable thresholds; these work well in practice:
            # - if there is *real* text, you'll usually get >300 chars/page.
            avg_chars_per_page = chars / sample_n if sample_n > 0 else 0
            if avg_chars_per_page < 200:
                logger.info("[OCR] PDF has low text density (%.0f chars/page avg) -> will use OCR", avg_chars_per_page)
                return None
            logger.info("[OCR] PDF has good text density (%.0f chars/page avg) -> digital extraction", avg_chars_per_page)
    except Exception as e:
        logger.warning("[OCR] Digital extraction check failed: %s", e)
        return None

    # Use canonical line extractor (words->lines + styles)
    try:
        from extraction.pdf_parser import parse_pdf_words_as_lines
        pages = parse_pdf_words_as_lines(path, run_layout_analysis=run_layout_analysis)
        if pages:
            # Convert blocks to lines for downstream line_comparison compatibility
            from extraction.line_extractor import extract_lines
            pages = extract_lines(pages)
        return pages if pages else None
    except Exception as e:
        logger.warning("[OCR] Digital extraction failed: %s -> falling back to OCR", e)
        return None
