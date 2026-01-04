"""
PDF Extraction Module - Unified Orchestrator
=============================================

This module provides the main entrypoint for PDF text and layout extraction.
It automatically detects document type (digital vs scanned) and routes to
the appropriate extraction method.

Architecture
------------

The extraction flow works as follows:

```
extract_pdf(path)
    │
    ├── Check settings (force_ocr, ocr_mode)
    │
    ├── Auto-detect document type
    │   └── _is_scanned_pdf() - samples first 5 pages for text content
    │
    └── Route to appropriate extractor:
        │
        ├── Digital PDF (has text layer)
        │   └── parse_pdf() → PyMuPDF text/style extraction
        │       └── analyze_layout() → YOLO-based table/figure detection
        │
        ├── Scanned PDF (image-based)
        │   └── ocr_pdf_multi() → OCR engine routing
        │       ├── DeepSeek-OCR (CUDA/MPS)
        │       ├── PaddleOCR (CPU fallback)
        │       └── Tesseract (legacy fallback)
        │
        └── Hybrid Mode
            ├── parse_pdf() → native extraction
            ├── ocr_pdf_multi() → OCR extraction
            └── _merge_extraction_results() → smart merge with safety gates
```

OCR Enhancement Modes
---------------------

- **auto** (default): Only use OCR for scanned PDFs
- **hybrid**: Run both native + OCR, merge results with hallucination protection
- **ocr_only**: Force OCR for all documents

Output Format
-------------

All extraction methods return `List[PageData]` with:
- page_num: 1-indexed page number
- width, height: Page dimensions in points
- blocks: List of TextBlock with text, bbox, and style
- metadata: Extraction method, layout info, OCR engine used

Example
-------
```python
from extraction import extract_pdf

# Auto-detect and extract
pages = extract_pdf("document.pdf")

# Force OCR mode
pages = extract_pdf("scanned_document.pdf", force_ocr=True)

# Access text
for page in pages:
    for block in page.blocks:
        print(f"Page {page.page_num}: {block.text}")
```
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from comparison.models import PageData
from config.settings import settings
from extraction.line_extractor import extract_document_lines, extract_lines, extract_lines_from_pages
from extraction.ocr_router import ocr_pdf_multi
from extraction.pdf_parser import parse_pdf, parse_pdf_words_as_lines
from utils.logging import logger

__all__ = ["extract_pdf", "extract_lines", "extract_lines_from_pages", "extract_document_lines", "parse_pdf", "parse_pdf_words_as_lines", "ocr_pdf_multi", "compute_text_quality_score"]


# Lithuanian letters for quality assessment
_LITHUANIAN_CHARS = set("ąčęėįšųūžĄČĘĖĮŠŲŪŽ")


def compute_text_quality_score(text: str) -> dict:
    """
    Compute text quality metrics for extraction quality assessment.
    
    This function evaluates the quality of extracted text to help decide
    whether OCR enhancement is needed or if the native extraction is sufficient.
    Particularly important for Lithuanian text with special characters.
    
    Metrics computed:
    - char_count: Total character count
    - word_count: Total word count
    - avg_word_length: Average word length (good text: 4-8)
    - lithuanian_ratio: Ratio of Lithuanian-specific chars (ąčęėįšųūž)
    - replacement_ratio: Ratio of replacement characters (�)
    - cid_ratio: Ratio of CID glyph patterns (font mapping failures)
    - quality_score: Overall quality score (0-1)
    
    Args:
        text: Extracted text to evaluate
    
    Returns:
        Dictionary with quality metrics and overall score
    
    Examples:
        >>> score = compute_text_quality_score("Ąžuolas ėjo į mokyklą")
        >>> score["quality_score"] > 0.8
        True
        
        >>> score = compute_text_quality_score("(cid:123)(cid:456)")
        >>> score["quality_score"] < 0.3
        True
    """
    if not text or len(text.strip()) < 10:
        return {
            "char_count": 0,
            "word_count": 0,
            "avg_word_length": 0.0,
            "lithuanian_ratio": 0.0,
            "replacement_ratio": 0.0,
            "cid_ratio": 0.0,
            "quality_score": 0.0,
        }
    
    stripped = text.strip()
    char_count = len(stripped)
    
    # Word analysis
    words = stripped.split()
    word_count = len(words)
    avg_word_length = sum(len(w) for w in words) / max(1, word_count)
    
    # Lithuanian character ratio (bonus for proper Lithuanian text)
    lt_count = sum(1 for c in stripped if c in _LITHUANIAN_CHARS)
    lithuanian_ratio = lt_count / char_count
    
    # Replacement character ratio (penalty for extraction failures)
    replacement_count = stripped.count("�") + stripped.count("\ufffd")
    replacement_ratio = replacement_count / char_count
    
    # CID glyph pattern ratio (font mapping failures)
    cid_matches = re.findall(r"\(cid:\d+\)", stripped)
    cid_ratio = (len(cid_matches) * 8) / char_count  # Each CID is ~8 chars
    
    # Non-printable character ratio
    non_ws = [ch for ch in stripped if not ch.isspace()]
    non_printable = sum(1 for ch in non_ws if not ch.isprintable())
    nonprintable_ratio = non_printable / max(1, len(non_ws))
    
    # Alphanumeric ratio (good text should have high ratio)
    alnum_count = sum(1 for ch in non_ws if ch.isalnum())
    alnum_ratio = alnum_count / max(1, len(non_ws))
    
    # Compute overall quality score
    quality_score = 1.0
    
    # Heavy penalties
    quality_score -= replacement_ratio * 5.0
    quality_score -= cid_ratio * 3.0
    quality_score -= nonprintable_ratio * 2.0
    
    # Moderate penalties/bonuses
    if alnum_ratio < 0.5:
        quality_score -= (0.5 - alnum_ratio)
    
    # Lithuanian character bonus (indicates proper Unicode handling)
    quality_score += lithuanian_ratio * 0.3
    
    # Word length bonus (reasonable word length indicates good extraction)
    if 3.0 <= avg_word_length <= 10.0:
        quality_score += 0.1
    
    # Clamp to [0, 1]
    quality_score = max(0.0, min(1.0, quality_score))
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "avg_word_length": avg_word_length,
        "lithuanian_ratio": lithuanian_ratio,
        "replacement_ratio": replacement_ratio,
        "cid_ratio": cid_ratio,
        "nonprintable_ratio": nonprintable_ratio,
        "alnum_ratio": alnum_ratio,
        "quality_score": quality_score,
    }


def _word_set(text: str) -> set[str]:
    """Normalize text to a set of words for lightweight overlap checks."""
    if not text:
        return set()
    # Lowercase, strip punctuation-ish, keep unicode letters/numbers/underscore.
    words = re.findall(r"\b[\w]+\b", text.lower())
    # Filter tiny tokens that add noise.
    return {w for w in words if len(w) >= 3}


def _looks_like_repetition(text: str) -> bool:
    """Cheap repetition/hallucination heuristic (model-agnostic)."""
    if not text or len(text) < 200:
        return False
    words = re.findall(r"\b[\w]+\b", text.lower())
    if len(words) < 50:
        return False
    counts: dict[str, int] = {}
    for w in words:
        if len(w) < 3:
            continue
        counts[w] = counts.get(w, 0) + 1
    if not counts:
        return False
    max_count = max(counts.values())
    # If the most common token dominates, likely a loop.
    if max_count / max(1, len(words)) > 0.35:
        return True
    # 6 identical tokens in a row → almost certainly repetition.
    for i in range(len(words) - 5):
        window = words[i : i + 6]
        if len(set(window)) == 1:
            return True
    return False


def _ocr_is_safe_to_replace_native(
    native_text: str,
    ocr_text: str,
    engine: str,
) -> tuple[bool, str, float]:
    """
    Decide whether OCR output is safe to replace native extraction in hybrid mode.

    Returns: (allowed, reason, overlap_ratio)
    """
    native_len = len(native_text or "")
    ocr_len = len(ocr_text or "")

    if native_len < 50:
        # If native is basically empty, allow OCR (that's the point of hybrid).
        return True, "native_empty", 1.0
    if ocr_len < 30:
        return False, "ocr_empty", 0.0

    native_words = _word_set(native_text)
    ocr_words = _word_set(ocr_text)
    if not native_words or not ocr_words:
        return False, "no_words", 0.0

    inter = len(native_words & ocr_words)
    overlap = inter / max(1, len(native_words))

    # Optional repetition rejection (helps especially for DeepSeek hallucinations)
    if settings.hybrid_ocr_reject_repetition and _looks_like_repetition(ocr_text):
        return False, "ocr_repetition", overlap

    # If OCR is wildly longer than native and overlap is low, treat as hallucination.
    if native_len > 200:
        length_ratio = ocr_len / max(1, native_len)
        if length_ratio >= settings.hybrid_ocr_max_length_ratio and overlap < settings.hybrid_ocr_min_word_overlap:
            return False, f"ocr_length_ratio_{length_ratio:.2f}_low_overlap", overlap

    if overlap < settings.hybrid_ocr_min_word_overlap:
        return False, "ocr_low_overlap", overlap

    return True, "ok", overlap


def extract_pdf(
    path: str | Path,
    force_ocr: bool = False,
    *,
    run_layout_analysis: bool = True,
    ocr_engine_priority: list[str] | None = None,
) -> List[PageData]:
    """
    Extract text and layout from a PDF, auto-detecting if it's digital or scanned.
    
    This is the main orchestrator function that:
    1. Validates the input PDF exists
    2. Checks OCR enhancement mode settings
    3. Auto-detects if PDF is scanned (image-based) or digital (has text layer)
    4. Routes to appropriate extraction method
    5. Runs layout analysis (tables/figures) if enabled
    6. Returns normalized PageData format regardless of extraction method
    
    Extraction Modes
    ----------------
    
    The behavior is controlled by `settings.ocr_enhancement_mode`:
    
    - **auto** (default): 
        - Digital PDFs → parse_pdf() with native text extraction
        - Scanned PDFs → ocr_pdf_multi() with OCR
    
    - **hybrid**:
        - Runs both native extraction AND OCR
        - Merges results with safety gate to prevent OCR hallucinations
        - Best for PDFs with mixed content (some native text, some images)
    
    - **ocr_only**:
        - Forces OCR for all documents
        - Use when native extraction quality is poor
    
    Args:
        path: Path to the PDF file (str or Path object)
        force_ocr: If True, always use OCR regardless of detection.
                   Overrides auto-detection and settings.
        run_layout_analysis: If True (default), runs YOLO-based layout detection
                            to identify tables and figures. Adds metadata to pages.
    
    Returns:
        List of PageData objects with:
        - page_num: 1-indexed page number
        - width, height: Page dimensions in points
        - blocks: List of TextBlock with text, bbox (normalized), style
        - metadata: Dict with extraction_method, layout info, etc.
    
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        RuntimeError: If required dependencies (PyMuPDF) are not installed
    
    Examples:
        # Basic extraction with auto-detection
        >>> pages = extract_pdf("document.pdf")
        >>> print(f"Extracted {len(pages)} pages")
        
        # Force OCR for a difficult document
        >>> pages = extract_pdf("scanned.pdf", force_ocr=True)
        
        # Skip layout analysis for faster processing
        >>> pages = extract_pdf("simple.pdf", run_layout_analysis=False)
    
    Notes:
        - Hybrid mode includes safety gates to prevent OCR from overwriting
          good native text with hallucinated content (especially from DeepSeek)
        - Layout analysis uses DocLayout-YOLO model for table/figure detection
        - All bbox coordinates are normalized to 0-1 range for consistent handling
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    logger.info("Extracting PDF: %s (force_ocr=%s, run_layout=%s)", path, force_ocr, run_layout_analysis)
    
    # Check OCR enhancement mode
    ocr_mode = getattr(settings, 'ocr_enhancement_mode', 'auto')
    use_ocr_for_all = getattr(settings, 'use_ocr_for_all_documents', False)
    
    if force_ocr or use_ocr_for_all or ocr_mode == 'ocr_only':
        logger.info("Using OCR mode (force_ocr=%s, use_ocr_for_all=%s, mode=%s)", 
                   force_ocr, use_ocr_for_all, ocr_mode)
        return ocr_pdf_multi(path, engine_priority=ocr_engine_priority, run_layout_analysis=run_layout_analysis)
    
    # Auto-detect: check if PDF has extractable text
    is_scanned = _is_scanned_pdf(path)
    
    if is_scanned:
        logger.info("Detected scanned PDF, using OCR")
        return ocr_pdf_multi(path, engine_priority=ocr_engine_priority, run_layout_analysis=run_layout_analysis)
    elif ocr_mode == 'hybrid':
        # Hybrid mode: extract with native parser, then enhance with OCR
        logger.info("Using hybrid mode: native extraction + OCR enhancement")
        native_pages = parse_pdf_words_as_lines(path, run_layout_analysis=run_layout_analysis)
        ocr_pages = ocr_pdf_multi(path, engine_priority=ocr_engine_priority, run_layout_analysis=run_layout_analysis)
        
        # Merge results: prefer OCR text for better accuracy, keep native formatting
        merged_pages = _merge_extraction_results(native_pages, ocr_pages)
        return merged_pages
    else:
        logger.info("Detected digital PDF, using word-level text extraction")
        return parse_pdf_words_as_lines(path, run_layout_analysis=run_layout_analysis)


def extract_lines(path: str | Path) -> List[PageData]:
    """
    Extract canonical line-level data from a PDF.

    Returns PageData with populated `lines` fields for each page.
    """
    path = Path(path)
    logger.info("Extracting lines: %s", path)
    return extract_document_lines(path)


def _merge_extraction_results(
    native_pages: List[PageData],
    ocr_pages: List[PageData],
) -> List[PageData]:
    """
    Merge native PDF extraction with OCR results.
    
    Combines the best of both: OCR text accuracy with native formatting/style info.
    
    Args:
        native_pages: Pages extracted using native PDF parsing
        ocr_pages: Pages extracted using OCR
    
    Returns:
        Merged PageData objects
    """
    merged = []
    
    # Create lookup for OCR pages
    ocr_lookup = {page.page_num: page for page in ocr_pages}
    
    for native_page in native_pages:
        merged_page = PageData(
            page_num=native_page.page_num,
            width=native_page.width,
            height=native_page.height,
            blocks=native_page.blocks.copy(),
            metadata=native_page.metadata.copy(),
        )
        
        # If OCR page exists for this page number, merge OCR text blocks
        if native_page.page_num in ocr_lookup:
            ocr_page = ocr_lookup[native_page.page_num]
            
            # Add OCR blocks (they may have better text accuracy)
            # Keep native blocks for formatting info, but prefer OCR text
            if ocr_page.blocks:
                # Merge strategy: use OCR text if native extraction has little/no text
                native_text_length = sum(len(b.text) for b in native_page.blocks)
                ocr_text_length = sum(len(b.text) for b in ocr_page.blocks)
                
                native_text = "\n".join(b.text for b in native_page.blocks if b.text)
                ocr_text = "\n".join(b.text for b in ocr_page.blocks if b.text)

                # Safety gate: prevent hallucinated OCR (esp. DeepSeek) from overwriting good native text.
                allowed, reason, overlap = _ocr_is_safe_to_replace_native(
                    native_text=native_text,
                    ocr_text=ocr_text,
                    engine=str(ocr_page.metadata.get("ocr_engine_used", settings.ocr_engine)),
                )
                merged_page.metadata["hybrid_ocr_overlap"] = overlap
                merged_page.metadata["hybrid_ocr_decision"] = reason

                if ocr_text_length > native_text_length * 1.2 and allowed:
                    logger.info(
                        "Page %d: hybrid OCR accepted (reason=%s overlap=%.2f), replacing native blocks (%d -> %d chars)",
                        native_page.page_num, reason, overlap, native_text_length, ocr_text_length
                    )
                    merged_page.blocks = ocr_page.blocks.copy()
                else:
                    # Native extraction is good (or OCR rejected). Keep native blocks.
                    logger.info(
                        "Page %d: hybrid OCR not used (reason=%s overlap=%.2f). Keeping native blocks (%d chars, OCR %d chars).",
                        native_page.page_num, reason, overlap, native_text_length, ocr_text_length
                    )
            
            # Merge metadata
            merged_page.metadata.update({
                "extraction_method": "hybrid",
                "native_blocks": len(native_page.blocks),
                "ocr_blocks": len(ocr_page.blocks),
            })
            merged_page.metadata.update(ocr_page.metadata)
        
        merged.append(merged_page)
    
    return merged


def _is_scanned_pdf(path: Path, threshold: float = 0.1) -> bool:
    """
    Detect if a PDF is scanned (image-based) by checking text content.
    
    Args:
        path: Path to PDF file
        threshold: Minimum ratio of pages with text to consider it digital
    
    Returns:
        True if PDF appears to be scanned (low text content)
    """
    def _text_layer_stats(text: str) -> dict:
        stripped = (text or "").strip()
        if not stripped:
            return {
                "char_count": 0,
                "non_ws_count": 0,
                "nonprintable_ratio": 1.0,
                "alnum_ratio": 0.0,
                "replacement_ratio": 0.0,
                "cid_ratio": 0.0,
            }

        non_ws = [ch for ch in stripped if not ch.isspace()]
        non_ws_count = len(non_ws)
        if non_ws_count == 0:
            return {
                "char_count": len(stripped),
                "non_ws_count": 0,
                "nonprintable_ratio": 1.0,
                "alnum_ratio": 0.0,
                "replacement_ratio": 0.0,
                "cid_ratio": 0.0,
            }

        non_printable = sum(1 for ch in non_ws if not ch.isprintable())
        nonprintable_ratio = non_printable / non_ws_count
        alnum_ratio = sum(1 for ch in non_ws if ch.isalnum()) / non_ws_count
        replacement = stripped.count("\ufffd") + stripped.count("�")
        replacement_ratio = replacement / non_ws_count

        # Common PDF extraction artifact when fonts are missing/mapped incorrectly.
        cid_hits = re.findall(r"\(cid:\d+\)", stripped)
        cid_ratio = (len(cid_hits) * 8) / max(1, len(stripped))  # rough normalized impact

        return {
            "char_count": len(stripped),
            "non_ws_count": non_ws_count,
            "nonprintable_ratio": nonprintable_ratio,
            "alnum_ratio": alnum_ratio,
            "replacement_ratio": replacement_ratio,
            "cid_ratio": cid_ratio,
        }

    def _looks_like_garbage_text(stats: dict) -> bool:
        # Treat as garbage only when we have enough characters; short text can be legitimate.
        if stats.get("char_count", 0) < 80:
            return False

        max_nonprint = float(getattr(settings, "scan_detection_garbage_max_nonprintable_ratio", 0.02))
        min_alnum = float(getattr(settings, "scan_detection_garbage_min_alnum_ratio", 0.12))

        if stats.get("replacement_ratio", 0.0) > 0.01:
            return True
        if stats.get("nonprintable_ratio", 0.0) > max_nonprint:
            return True
        if stats.get("cid_ratio", 0.0) > 0.15:
            return True

        # If the text is long but barely has letters/numbers, it's likely garbage.
        if stats.get("alnum_ratio", 0.0) < min_alnum:
            return True

        return False

    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not available, assuming digital PDF")
        return False
    
    doc = fitz.open(path)
    pages_with_text = 0
    total_pages = len(doc)
    
    # Sample first few pages to determine type
    sample_target = int(getattr(settings, "scan_detection_sample_pages", 5))
    if sample_target <= 0:
        sample_target = 5
    sample_size = min(sample_target, total_pages)

    min_chars = int(getattr(settings, "scan_detection_page_text_min_chars", 50))
    if min_chars <= 0:
        min_chars = 50
    for i in range(sample_size):
        page = doc[i]
        text = page.get_text("text")
        stats = _text_layer_stats(text)
        is_good = stats["char_count"] >= min_chars and not _looks_like_garbage_text(stats)
        if is_good:
            pages_with_text += 1
    
    doc.close()
    
    text_ratio = pages_with_text / sample_size if sample_size > 0 else 0

    # Backward-compatible threshold parameter, but prefer settings default.
    min_good_ratio = float(getattr(settings, "scan_detection_min_good_pages_ratio", threshold))
    if min_good_ratio < 0:
        min_good_ratio = 0.0
    if min_good_ratio > 1:
        min_good_ratio = 1.0

    is_scanned = text_ratio < min_good_ratio
    
    logger.debug(
        "PDF detection: %d/%d pages have usable text (ratio=%.2f, min_ratio=%.2f), scanned=%s",
        pages_with_text,
        sample_size,
        text_ratio,
        min_good_ratio,
        is_scanned,
    )
    
    return is_scanned
