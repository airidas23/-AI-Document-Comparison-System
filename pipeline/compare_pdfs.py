"""
Main orchestrator: end-to-end PDF comparison pipeline.

Provides a single entrypoint that:
1. Opens PDF files
2. Auto-detects digital vs scanned
3. Runs extraction (OCR if needed) + layout analysis
4. Aligns pages, computes diffs
5. Returns a unified DiffReport / ComparisonResult
6. Optionally exports debug JSON for analysis

Phase 2 additions:
- Extraction fingerprint (extraction_manifest.json)
- Normalization config tracking
- Detailed timing breakdown
- Quality metrics (phantom diff tracking)
- OCR-aware change detection (two-stage gating)
- Per-engine calibration profiles
- Enhanced OCR quality metrics (precision proxy, severity breakdown)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from comparison.alignment import align_pages
from comparison.diff_classifier import classify_diffs, get_diff_summary
from comparison.diff_fusion import fuse_diffs
from comparison.figure_comparison import compare_figure_captions
from comparison.formatting_comparison import compare_formatting
from comparison.line_comparison import compare_lines
from comparison.models import ComparisonResult, PageData, ComparisonReport, Diff
from comparison.table_comparison import compare_tables
from comparison.text_comparison import TextComparator
from comparison.text_normalizer import (
    NormalizationConfig,
)
from config.settings import settings
from extraction import extract_pdf
from extraction.line_extractor import extract_lines  # Use new function that reuses blocks
from extraction.header_footer_detector import compare_headers_footers
from extraction.layout_analyzer import analyze_layout
from utils.logging import logger
from utils.performance import track_time
from utils.text_normalization import normalize_text_full

# Phase 2 OCR-aware imports
try:
    from utils.ocr_quality_metrics import OCRQualityMetrics
    HAS_OCR_QUALITY_METRICS = True
except ImportError:
    HAS_OCR_QUALITY_METRICS = False
    OCRQualityMetrics = None

HAS_OCR_GATING = importlib.util.find_spec("comparison.ocr_gating") is not None
HAS_ENGINE_PROFILES = importlib.util.find_spec("config.engine_profiles") is not None
HAS_PAGE_CHECKSUM = importlib.util.find_spec("utils.page_checksum") is not None

try:
    from utils.ocr_normalizer import normalize_ocr_compare, normalize_ocr_strict
    HAS_OCR_NORMALIZER = True
except ImportError:
    HAS_OCR_NORMALIZER = False


@dataclass
class PipelineConfig:
    """Configuration for the comparison pipeline."""
    
    # OCR settings
    ocr_mode: Literal["auto", "hybrid", "ocr_only"] = "auto"
    ocr_engine: Optional[str] = None  # None = use settings default
    force_ocr: bool = False
    
    # Layout settings
    run_layout_analysis: bool = True
    
    # Comparison settings
    sensitivity_threshold: Optional[float] = None  # None = use settings default
    
    # Fusion settings
    use_fusion: bool = True  # Enable diff fusion (triangulation)
    fusion_strategy: Literal["triangulation", "union", "intersection"] = "triangulation"
    fusion_iou_threshold: float = 0.3  # Minimum IoU to consider diffs overlapping
    
    # Debug and export settings
    debug_mode: bool = False  # Enable debug JSON export
    debug_output_path: Optional[str] = None  # Override default debug path
    export_comparison_report: bool = True  # Generate ComparisonReport
    
    # Performance settings
    profile: bool = False
    
    def apply_to_settings(self) -> None:
        """Apply config to global settings."""
        if self.ocr_engine:
            settings.ocr_engine = self.ocr_engine
        if self.sensitivity_threshold is not None:
            settings.text_similarity_threshold = self.sensitivity_threshold
        settings.ocr_enhancement_mode = self.ocr_mode
        if self.ocr_mode == "ocr_only" or self.force_ocr:
            settings.use_ocr_for_all_documents = True
        else:
            settings.use_ocr_for_all_documents = False


def _bbox_intersection_area(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax0 = float(a.get("x", 0.0))
    ay0 = float(a.get("y", 0.0))
    ax1 = ax0 + float(a.get("width", 0.0))
    ay1 = ay0 + float(a.get("height", 0.0))

    bx0 = float(b.get("x", 0.0))
    by0 = float(b.get("y", 0.0))
    bx1 = bx0 + float(b.get("width", 0.0))
    by1 = by0 + float(b.get("height", 0.0))

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = ix1 - ix0
    ih = iy1 - iy0
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    return iw * ih


def _filter_lines_overlapping_regions(
    pages: List[PageData],
    *,
    region_key: str,
    overlap_ratio: float = 0.5,
) -> List[PageData]:
    """Return shallow-copied pages with lines removed if they overlap tagged regions.

    Used to avoid double-reporting text changes inside tables/figures where
    specialized comparators already emit diffs.
    """
    out: List[PageData] = []
    for page in pages:
        regions = (page.metadata or {}).get(region_key) or []
        bboxes = []
        for r in regions:
            if isinstance(r, dict) and isinstance(r.get("bbox"), dict):
                bboxes.append(r["bbox"])
        if not bboxes or not page.lines:
            out.append(page)
            continue

        kept = []
        for ln in page.lines:
            lb = getattr(ln, "bbox", None)
            if not isinstance(lb, dict):
                kept.append(ln)
                continue
            line_area = float(lb.get("width", 0.0)) * float(lb.get("height", 0.0))
            if line_area <= 0.0:
                kept.append(ln)
                continue
            max_overlap = 0.0
            for rb in bboxes:
                ov = _bbox_intersection_area(lb, rb)
                if ov > max_overlap:
                    max_overlap = ov
            if (max_overlap / line_area) < overlap_ratio:
                kept.append(ln)

        out.append(
            PageData(
                page_num=page.page_num,
                width=page.width,
                height=page.height,
                blocks=page.blocks,
                lines=kept,
                metadata=page.metadata,
            )
        )
    return out


# =============================================================================
# Extraction Fingerprint (Phase 2 - Step 0)
# =============================================================================

@dataclass
class ExtractionManifest:
    """Extraction fingerprint for a document.
    
    Captures all extraction parameters to ensure reproducibility
    and detect when re-extraction is needed.
    """
    # Input identification
    input_path: str = ""
    input_hash: str = ""  # SHA-256 of file content
    
    # Detection results
    input_type_detected: str = ""  # "digital", "scanned", "hybrid"
    
    # Extraction parameters
    engine_used: str = ""  # "native", "tesseract", "paddle", "deepseek"
    force_ocr: bool = False
    render_dpi: int = 0
    granularity: str = ""  # "word", "line", "block"
    
    # Output statistics
    page_count: int = 0
    blocks_count_total: int = 0
    lines_count_total: int = 0
    chars_count_total: int = 0
    
    # Confidence statistics
    avg_conf: float = 0.0
    conf_p10: float = 0.0
    conf_p50: float = 0.0
    conf_p90: float = 0.0
    
    # Per-page metadata
    page_manifests: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Export to JSON-serializable dict."""
        return asdict(self)
    
    @classmethod
    def from_pages(
        cls,
        pages: List[PageData],
        input_path: str,
        engine_used: str,
        force_ocr: bool = False,
        render_dpi: int = 0,
    ) -> "ExtractionManifest":
        """Build manifest from extracted pages."""
        manifest = cls(
            input_path=str(input_path),
            engine_used=engine_used,
            force_ocr=force_ocr,
            render_dpi=render_dpi,
            page_count=len(pages),
        )
        
        # Compute input hash if file exists
        path = Path(input_path)
        if path.exists():
            with open(path, "rb") as f:
                manifest.input_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Aggregate statistics
        all_confidences: List[float] = []
        page_manifests: List[Dict[str, Any]] = []
        
        for page in pages:
            md = page.metadata or {}
            
            # Count blocks and lines
            blocks_count = len(page.blocks) if page.blocks else 0
            lines_count = len(page.lines) if page.lines else 0
            chars_count = sum(len(b.text or "") for b in (page.blocks or []))
            
            manifest.blocks_count_total += blocks_count
            manifest.lines_count_total += lines_count
            manifest.chars_count_total += chars_count
            
            # Collect confidence values
            for block in (page.blocks or []):
                if block.confidence and block.confidence > 0:
                    all_confidences.append(block.confidence)
            
            # Detect input type from metadata
            extraction_method = str(md.get("extraction_method", ""))
            if "ocr" in extraction_method.lower():
                page_type = "scanned"
            else:
                page_type = "digital"
            
            if not manifest.input_type_detected:
                manifest.input_type_detected = page_type
            elif manifest.input_type_detected != page_type:
                manifest.input_type_detected = "hybrid"
            
            # Get granularity
            if not manifest.granularity:
                manifest.granularity = md.get("granularity", "block")
            
            # Per-page manifest
            page_manifests.append({
                "page_num": page.page_num,
                "width": page.width,
                "height": page.height,
                "blocks_count": blocks_count,
                "lines_count": lines_count,
                "chars_count": chars_count,
                "extraction_method": extraction_method,
                "ocr_engine": md.get("ocr_engine_used", ""),
                "ocr_policy": md.get("ocr_policy", ""),
                "ocr_engine_selected": md.get("ocr_engine_selected", ""),
                "ocr_status": md.get("ocr_status", ""),
                "ocr_failure_reason": md.get("ocr_failure_reason", ""),
                "ocr_attempts": md.get("ocr_attempts", []),
            })
        
        manifest.page_manifests = page_manifests
        
        # Compute confidence percentiles
        if all_confidences:
            sorted_conf = sorted(all_confidences)
            n = len(sorted_conf)
            manifest.avg_conf = sum(sorted_conf) / n
            manifest.conf_p10 = sorted_conf[int(n * 0.10)] if n > 10 else sorted_conf[0]
            manifest.conf_p50 = sorted_conf[int(n * 0.50)]
            manifest.conf_p90 = sorted_conf[int(n * 0.90)] if n > 10 else sorted_conf[-1]
        
        return manifest


# =============================================================================
# Timing Breakdown (Phase 2 - Step 7)
# =============================================================================


@dataclass
class PipelineMetrics:
    """Performance metrics from pipeline execution."""
    total_time: float = 0.0
    extraction_time: float = 0.0
    layout_time: float = 0.0
    comparison_time: float = 0.0
    classification_time: float = 0.0
    fusion_time: float = 0.0
    
    # Phase 2: Detailed timing breakdown
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    
    pages_processed: int = 0
    diffs_found: int = 0
    diffs_before_fusion: int = 0
    diffs_after_fusion: int = 0
    
    # Phase 2: Quality metrics
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def time_per_page(self) -> float:
        return self.total_time / max(1, self.pages_processed)
    
    @property
    def fusion_reduction_percent(self) -> float:
        if self.diffs_before_fusion == 0:
            return 0.0
        return (1 - self.diffs_after_fusion / self.diffs_before_fusion) * 100
    
    def to_dict(self) -> dict:
        """Export to JSON-serializable dict."""
        return {
            "total_time": self.total_time,
            "extraction_time": self.extraction_time,
            "layout_time": self.layout_time,
            "comparison_time": self.comparison_time,
            "classification_time": self.classification_time,
            "fusion_time": self.fusion_time,
            "timing_breakdown": self.timing_breakdown,
            "pages_processed": self.pages_processed,
            "diffs_found": self.diffs_found,
            "diffs_before_fusion": self.diffs_before_fusion,
            "diffs_after_fusion": self.diffs_after_fusion,
            "time_per_page": self.time_per_page,
            "fusion_reduction_percent": self.fusion_reduction_percent,
            "quality_metrics": self.quality_metrics,
        }


# =============================================================================
# Quality Metrics (Phase 2 - Step 7)
# =============================================================================

@dataclass
class QualityMetrics:
    """Quality metrics for comparison results.
    
    Tracks phantom diff indicators and comparison stability.
    """
    # Diff counts by type
    content_diffs: int = 0
    formatting_diffs: int = 0
    layout_diffs: int = 0
    visual_diffs: int = 0
    
    # Phantom diff indicators
    whitespace_only_diffs: int = 0
    formatting_only_diffs: int = 0
    low_confidence_diffs: int = 0  # confidence < 0.5
    
    # Alignment quality
    unmatched_blocks_a: int = 0
    unmatched_blocks_b: int = 0
    
    # Early termination stats
    pages_skipped_identical: int = 0
    blocks_skipped_identical: int = 0
    
    # Candidate generation stats (Step 2)
    skipped_candidates_count: int = 0
    fallback_relaxations_count: int = 0
    
    @property
    def whitespace_only_ratio(self) -> float:
        """Ratio of whitespace-only diffs (should decrease with optimization)."""
        total = self.content_diffs + self.formatting_diffs + self.layout_diffs
        return self.whitespace_only_diffs / max(1, total)
    
    @property
    def phantom_diff_proxy(self) -> float:
        """Proxy metric for phantom diffs (lower is better)."""
        total = self.content_diffs + self.formatting_diffs + self.layout_diffs
        phantom = self.whitespace_only_diffs + self.formatting_only_diffs + self.low_confidence_diffs
        return phantom / max(1, total)
    
    def to_dict(self) -> dict:
        """Export to JSON-serializable dict."""
        return {
            "content_diffs": self.content_diffs,
            "formatting_diffs": self.formatting_diffs,
            "layout_diffs": self.layout_diffs,
            "visual_diffs": self.visual_diffs,
            "whitespace_only_diffs": self.whitespace_only_diffs,
            "formatting_only_diffs": self.formatting_only_diffs,
            "low_confidence_diffs": self.low_confidence_diffs,
            "unmatched_blocks_a": self.unmatched_blocks_a,
            "unmatched_blocks_b": self.unmatched_blocks_b,
            "pages_skipped_identical": self.pages_skipped_identical,
            "blocks_skipped_identical": self.blocks_skipped_identical,
            "skipped_candidates_count": self.skipped_candidates_count,
            "fallback_relaxations_count": self.fallback_relaxations_count,
            "whitespace_only_ratio": self.whitespace_only_ratio,
            "phantom_diff_proxy": self.phantom_diff_proxy,
        }


# =============================================================================
# Phase 2: Build Extraction Manifest
# =============================================================================

def _build_extraction_manifest(
    pages: List[PageData],
    input_hash: str,
    ocr_engine: Optional[str] = None,
) -> ExtractionManifest:
    """Build extraction manifest from extracted pages.
    
    Args:
        pages: List of extracted pages
        input_hash: SHA256 hash of input file
        ocr_engine: OCR engine name if used
        
    Returns:
        ExtractionManifest with extraction fingerprint
    """
    manifest = ExtractionManifest(
        input_hash=input_hash,
        engine_used=ocr_engine or "native",
        page_count=len(pages),
    )
    
    all_confidences: List[float] = []
    page_manifests: List[Dict[str, Any]] = []
    
    for page in pages:
        md = page.metadata or {}
        
        # Count blocks and lines
        blocks_count = len(page.blocks) if page.blocks else 0
        lines_count = len(page.lines) if page.lines else 0
        chars_count = sum(len(b.text or "") for b in (page.blocks or []))
        
        manifest.blocks_count_total += blocks_count
        manifest.lines_count_total += lines_count
        manifest.chars_count_total += chars_count
        
        # Collect confidence values
        for block in (page.blocks or []):
            confidence = getattr(block, "confidence", None)
            if confidence is None:
                block_md = getattr(block, "metadata", None) or {}
                confidence = block_md.get("confidence")

            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and confidence > 0:
                all_confidences.append(float(confidence))
        
        # Detect input type from metadata
        extraction_method = str(md.get("extraction_method", ""))
        if "ocr" in extraction_method.lower():
            page_type = "scanned"
        else:
            page_type = "digital"
        
        if not manifest.input_type_detected:
            manifest.input_type_detected = page_type
        elif manifest.input_type_detected != page_type:
            manifest.input_type_detected = "hybrid"
        
        # Get granularity
        if not manifest.granularity:
            manifest.granularity = md.get("granularity", "block")
        
        # Per-page manifest
        page_manifests.append({
            "page_num": page.page_num,
            "width": page.width,
            "height": page.height,
            "blocks_count": blocks_count,
            "lines_count": lines_count,
            "chars_count": chars_count,
            "extraction_method": extraction_method,
            "ocr_engine": md.get("ocr_engine_used", ""),
            "ocr_policy": md.get("ocr_policy", ""),
            "ocr_engine_selected": md.get("ocr_engine_selected", ""),
            "ocr_status": md.get("ocr_status", ""),
            "ocr_failure_reason": md.get("ocr_failure_reason", ""),
            "ocr_attempts": md.get("ocr_attempts", []),
        })
    
    manifest.page_manifests = page_manifests
    
    # Compute confidence percentiles
    if all_confidences:
        sorted_conf = sorted(all_confidences)
        n = len(sorted_conf)
        manifest.avg_conf = sum(sorted_conf) / n
        manifest.conf_p10 = sorted_conf[int(n * 0.10)] if n > 10 else sorted_conf[0]
        manifest.conf_p50 = sorted_conf[int(n * 0.50)]
        manifest.conf_p90 = sorted_conf[int(n * 0.90)] if n > 10 else sorted_conf[-1]
    
    return manifest


# =============================================================================
# Phase 2: Page Checksum for Early Termination (Step 4)
# =============================================================================

def _compute_page_checksum(page: PageData, is_ocr: bool = False) -> str:
    """Compute a content checksum for a page for early termination.
    
    The checksum is based on normalized text content.

    IMPORTANT: For digital (non-OCR) pages we also include a lightweight
    formatting/layout signature (block style + quantized bbox) so that
    formatting-only changes (e.g., font size, spacing, table padding/borders)
    are NOT incorrectly treated as "identical" and skipped.
    """
    # Use enhanced OCR normalizer if available
    if HAS_OCR_NORMALIZER and is_ocr:
        texts = []
        for block in page.blocks:
            if block.text and block.text.strip():
                normalized = normalize_ocr_strict(block.text.strip())
                if normalized:
                    texts.append(normalized)
        content = "\n".join(texts)
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
    
    # Digital/non-OCR: include text + (style + bbox) signature.
    def _q(v: float) -> float:
        # Quantize to reduce checksum churn from tiny float noise.
        return round(float(v), 1)

    texts: List[str] = []
    for block in page.blocks:
        raw = (block.text or "").strip()
        if not raw:
            continue

        normalized = normalize_text_full(raw)
        if not normalized:
            continue

        st = getattr(block, "style", None)
        font = getattr(st, "font", None) or ""
        size = getattr(st, "size", None)
        bold = bool(getattr(st, "bold", False))
        italic = bool(getattr(st, "italic", False))
        color = getattr(st, "color", None)

        bb = getattr(block, "bbox", None) or {}
        bx = _q(bb.get("x", 0.0))
        by = _q(bb.get("y", 0.0))
        bw = _q(bb.get("width", 0.0))
        bh = _q(bb.get("height", 0.0))

        texts.append(
            "|".join(
                [
                    normalized,
                    str(font),
                    f"{float(size):.1f}" if isinstance(size, (int, float)) else "",
                    "b" if bold else "r",
                    "i" if italic else "n",
                    ",".join(str(int(c)) for c in color) if isinstance(color, tuple) and len(color) == 3 else "",
                    f"{bx},{by},{bw},{bh}",
                ]
            )
        )

    content = "\n".join(texts)
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


def _pages_are_identical(page_a: PageData, page_b: PageData, is_ocr: bool = False) -> bool:
    """Check if two pages are content-identical using checksums."""
    checksum_a = _compute_page_checksum(page_a, is_ocr)
    checksum_b = _compute_page_checksum(page_b, is_ocr)
    return checksum_a == checksum_b


def _detect_ocr_pages(pages: List[PageData]) -> tuple[bool, set[int]]:
    """
    Detect which pages were extracted using OCR.
    
    Returns:
        Tuple of (any_ocr: bool, ocr_page_nums: set of page numbers that used OCR)
    """
    ocr_page_nums: set[int] = set()
    
    for page in pages:
        md = page.metadata or {}
        extraction_method = str(md.get("extraction_method") or "")
        line_extraction_method = str(md.get("line_extraction_method") or "")
        ocr_engine_used = str(md.get("ocr_engine_used") or "")
        
        is_ocr = (
            "ocr" in extraction_method.lower()
            or "ocr" in line_extraction_method.lower()
            or "ocr" in ocr_engine_used.lower()
            or "tesseract" in ocr_engine_used.lower()
            or "paddle" in ocr_engine_used.lower()
            or "deepseek" in ocr_engine_used.lower()
        )
        
        if is_ocr:
            ocr_page_nums.add(page.page_num)
    
    return bool(ocr_page_nums), ocr_page_nums


class ComparisonPipeline:
    """
    End-to-end document comparison pipeline.
    
    Usage:
        pipeline = ComparisonPipeline(config)
        result = pipeline.compare(pdf_a, pdf_b)
        
        # Or step-by-step:
        pages_a = pipeline.extract(pdf_a)
        pages_b = pipeline.extract(pdf_b)
        result = pipeline.diff(pages_a, pages_b, pdf_a, pdf_b)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.metrics = PipelineMetrics()
        self.quality_metrics = QualityMetrics()
        self._text_comparator = TextComparator()
        self._comparison_report: Optional[ComparisonReport] = None
        
        # Phase 2: Extraction manifests
        self._manifest_a: Optional[ExtractionManifest] = None
        self._manifest_b: Optional[ExtractionManifest] = None
        
        # Phase 2: Normalization config
        self._normalization_config = NormalizationConfig.default_digital()
    
    def compare(
        self,
        pdf_a: str | Path,
        pdf_b: str | Path,
    ) -> ComparisonResult:
        """
        Full end-to-end comparison of two PDF documents.
        
        Args:
            pdf_a: Path to first PDF
            pdf_b: Path to second PDF
            
        Returns:
            ComparisonResult with all diffs and metadata
        """
        start_time = time.time()
        pdf_a = Path(pdf_a)
        pdf_b = Path(pdf_b)
        
        logger.info("=== Starting comparison pipeline ===")
        logger.info("Doc A: %s", pdf_a)
        logger.info("Doc B: %s", pdf_b)
        
        # Apply config
        self.config.apply_to_settings()
        
        # Extract both documents with Phase 2 manifest tracking
        pages_a = self.extract(pdf_a, is_doc_a=True)
        pages_b = self.extract(pdf_b, is_doc_a=False)
        
        # Run comparison
        result = self.diff(pages_a, pages_b, str(pdf_a), str(pdf_b))
        
        self.metrics.total_time = time.time() - start_time
        self.metrics.pages_processed = max(len(pages_a), len(pages_b))
        self.metrics.diffs_found = len(result.diffs)
        
        logger.info("=== Pipeline complete ===")
        logger.info("Time: %.2fs (%.2fs/page)", 
                   self.metrics.total_time, 
                   self.metrics.time_per_page)
        logger.info("Diffs: %d", self.metrics.diffs_found)
        
        # Export debug JSON if enabled
        if self.config.debug_mode or getattr(settings, 'debug_mode', False):
            self._export_debug_json(result, pdf_a, pdf_b)
        
        return result
    
    def _export_debug_json(
        self,
        result: ComparisonResult,
        pdf_a: Path,
        pdf_b: Path,
    ) -> None:
        """Export detailed debug JSON for analysis (Phase 2 enhanced)."""
        debug_path = (
            self.config.debug_output_path 
            or getattr(settings, 'debug_output_path', './debug_output')
        )
        debug_dir = Path(debug_path)
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_debug_{timestamp}.json"
        
        # Build debug report with Phase 2 additions
        debug_data = {
            "timestamp": timestamp,
            "doc_a": str(pdf_a),
            "doc_b": str(pdf_b),
            
            # Phase 2: Metrics with timing breakdown
            "metrics": self.metrics.to_dict(),
            
            # Phase 2: Quality metrics
            "quality_metrics": self.quality_metrics.to_dict(),
            
            "config": {
                "ocr_mode": self.config.ocr_mode,
                "ocr_engine": self.config.ocr_engine,
                "force_ocr": self.config.force_ocr,
                "sensitivity_threshold": self.config.sensitivity_threshold,
                "fusion_strategy": self.config.fusion_strategy,
            },
            
            # Phase 2: Extraction manifests
            "extraction_manifest_a": self._manifest_a.to_dict() if self._manifest_a else None,
            "extraction_manifest_b": self._manifest_b.to_dict() if self._manifest_b else None,
            
            # Phase 2: Normalization config
            "normalization_config": self._normalization_config.to_dict(),
            
            "summary": result.summary,
            "diffs": [self._diff_to_dict(d) for d in result.diffs],
            "comparison_report": self._build_comparison_report_dict(result),
        }
        
        output_file = debug_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("Debug JSON exported to: %s", output_file)
    
    def _diff_to_dict(self, diff: Diff) -> dict:
        """Convert Diff object to serializable dict."""
        return {
            "page_num": diff.page_num,
            "page_num_b": getattr(diff, 'page_num_b', None),
            "diff_type": diff.diff_type,
            "change_type": diff.change_type,
            "old_text": diff.old_text,
            "new_text": diff.new_text,
            "old_text_normalized": normalize_text_full(diff.old_text) if diff.old_text else None,
            "new_text_normalized": normalize_text_full(diff.new_text) if diff.new_text else None,
            "bbox": diff.bbox,
            "bbox_b": getattr(diff, 'bbox_b', None),
            "confidence": diff.confidence,
            "element_type": getattr(diff, 'element_type', None),
            "layout_change_type": getattr(diff, 'layout_change_type', None),
            "sources": getattr(diff, 'sources', None),
            "word_diffs": [
                {"old": wd.old_word, "new": wd.new_word, "type": wd.change_type}
                for wd in (getattr(diff, 'word_diffs', None) or [])
            ],
            "metadata": diff.metadata,
        }
    
    def _build_comparison_report_dict(self, result: ComparisonResult) -> dict:
        """Build ComparisonReport-style dict for debug output."""
        # Group diffs by type
        text_diffs = [d for d in result.diffs if d.metadata.get("type") not in ("table", "figure", "formula")]
        table_diffs = [d for d in result.diffs if d.metadata.get("type") == "table"]
        figure_diffs = [d for d in result.diffs if d.metadata.get("type") == "figure"]
        formula_diffs = [d for d in result.diffs if d.metadata.get("type") == "formula"]
        layout_diffs = [d for d in result.diffs if d.change_type == "layout"]
        
        return {
            "text_diffs_count": len(text_diffs),
            "table_diffs_count": len(table_diffs),
            "figure_diffs_count": len(figure_diffs),
            "formula_diffs_count": len(formula_diffs),
            "layout_diffs_count": len(layout_diffs),
            "total_diffs": len(result.diffs),
            "diffs_before_fusion": self.metrics.diffs_before_fusion,
            "diffs_after_fusion": self.metrics.diffs_after_fusion,
            "fusion_reduction_percent": self.metrics.fusion_reduction_percent,
            "by_change_type": {
                "content": len([d for d in result.diffs if d.change_type == "content"]),
                "visual": len([d for d in result.diffs if d.change_type == "visual"]),
                "layout": len([d for d in result.diffs if d.change_type == "layout"]),
                "formatting": len([d for d in result.diffs if d.change_type == "formatting"]),
            },
            "by_diff_type": {
                "added": len([d for d in result.diffs if d.diff_type == "added"]),
                "deleted": len([d for d in result.diffs if d.diff_type == "deleted"]),
                "modified": len([d for d in result.diffs if d.diff_type == "modified"]),
            },
        }
    
    def get_comparison_report(self) -> Optional[ComparisonReport]:
        """Get the last generated ComparisonReport."""
        return self._comparison_report
    
    def extract(
        self, 
        pdf_path: str | Path,
        is_doc_a: bool = True,
    ) -> List[PageData]:
        """
        Extract pages from a single PDF with optional layout analysis.
        
        Args:
            pdf_path: Path to PDF file
            is_doc_a: Whether this is doc A (for manifest tracking)
            
        Returns:
            List of PageData with text blocks and layout metadata
        """
        pdf_path = Path(pdf_path)
        logger.info("Extracting: %s", pdf_path)
        
        extract_start = time.time()
        
        # Phase 2: Compute input hash for extraction fingerprint
        input_hash = self._compute_file_hash(pdf_path)
        
        # Run extraction (auto-detects digital vs scanned)
        pages = extract_pdf(pdf_path, force_ocr=self.config.force_ocr)
        
        extraction_time = time.time() - extract_start
        self.metrics.extraction_time += extraction_time
        self.metrics.timing_breakdown[f"extraction_{'a' if is_doc_a else 'b'}"] = extraction_time
        
        # Convert blocks to lines WITHOUT re-running OCR
        # This reuses already-extracted blocks from extract_pdf
        line_start = time.time()
        try:
            pages = extract_lines(pages)  # Reuses blocks, no additional OCR
        except Exception as exc:
            logger.warning("Line extraction failed: %s. Continuing without lines.", exc)
        self.metrics.timing_breakdown[f"line_extraction_{'a' if is_doc_a else 'b'}"] = time.time() - line_start

        # Ensure layout analysis is run for all pages
        # (digital PDFs already have it via pdf_parser, but OCR pages might not)
        layout_start = time.time()
        if self.config.run_layout_analysis:
            pages = self._ensure_layout_metadata(pdf_path, pages)
        self.metrics.timing_breakdown[f"layout_{'a' if is_doc_a else 'b'}"] = time.time() - layout_start
        
        # Phase 2: Build extraction manifest
        manifest = _build_extraction_manifest(pages, input_hash, self.config.ocr_engine)
        if is_doc_a:
            self._manifest_a = manifest
        else:
            self._manifest_b = manifest
        
        logger.info("Extracted %d pages in %.2fs", len(pages), extraction_time)
        return pages
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file for extraction fingerprinting."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]  # First 16 chars for brevity

    def _merge_line_data(self, pages: List[PageData], line_pages: List[PageData]) -> None:
        """Merge line-level extraction output into existing PageData."""
        line_lookup = {page.page_num: page for page in line_pages}
        for page in pages:
            line_page = line_lookup.get(page.page_num)
            if not line_page:
                continue
            page.lines = line_page.lines
            if line_page.metadata:
                page.metadata.update(line_page.metadata)
    
    def _ensure_layout_metadata(
        self, 
        pdf_path: Path, 
        pages: List[PageData]
    ) -> List[PageData]:
        """
        Ensure all pages have layout metadata (tables, figures).
        
        If layout_analyzed is missing, run layout analyzer.
        This is especially important for OCR-extracted pages.
        """
        # Check if any page is missing layout analysis
        needs_layout = any(
            not page.metadata.get("layout_analyzed", False) 
            for page in pages
        )
        
        if not needs_layout:
            logger.debug("All pages already have layout metadata")
            return pages
        
        logger.info("Running layout analysis for pages missing metadata")
        layout_start = time.time()
        
        try:
            layout_pages = analyze_layout(pdf_path)
            layout_lookup = {p.page_num: p for p in layout_pages}
            
            # Merge layout metadata into pages
            for page in pages:
                if page.page_num in layout_lookup and not page.metadata.get("layout_analyzed"):
                    layout_page = layout_lookup[page.page_num]
                    page.metadata.update({
                        "tables": layout_page.metadata.get("tables", []),
                        "figures": layout_page.metadata.get("figures", []),
                        "text_blocks": layout_page.metadata.get("text_blocks", []),
                        "layout_analyzed": True,
                        "layout_method": layout_page.metadata.get("layout_method", "yolo"),
                    })
            
            layout_time = time.time() - layout_start
            self.metrics.layout_time += layout_time
            logger.info("Layout analysis completed in %.2fs", layout_time)
            
        except Exception as e:
            logger.warning("Layout analysis failed: %s. Continuing without.", e)
            # Mark as analyzed (even if failed) to avoid retrying
            for page in pages:
                if not page.metadata.get("layout_analyzed"):
                    page.metadata["layout_analyzed"] = False
                    page.metadata["layout_error"] = str(e)
        
        return pages
    
    def diff(
        self,
        pages_a: List[PageData],
        pages_b: List[PageData],
        doc1_path: str,
        doc2_path: str,
    ) -> ComparisonResult:
        """
        Compare two sets of pages and return classified diffs.
        
        Args:
            pages_a: Pages from first document
            pages_b: Pages from second document
            doc1_path: Path to doc1 (for result metadata)
            doc2_path: Path to doc2 (for result metadata)
            
        Returns:
            ComparisonResult with all diffs
        """
        logger.info("Comparing %d vs %d pages", len(pages_a), len(pages_b))
        compare_start = time.time()
        
        # =================================================================
        # CRITICAL: Detect OCR pages early to control downstream modules
        # OCR pages have unreliable font/size/spacing data, so we skip or
        # downweight formatting/layout comparisons for them.
        # =================================================================
        any_ocr_a, ocr_pages_a = _detect_ocr_pages(pages_a)
        any_ocr_b, ocr_pages_b = _detect_ocr_pages(pages_b)
        any_ocr = any_ocr_a or any_ocr_b
        
        if any_ocr:
            logger.info(
                "OCR detected: %d pages in doc A, %d pages in doc B",
                len(ocr_pages_a), len(ocr_pages_b)
            )
        
        # Collect diffs from all comparison modules
        # Each module returns a list of diffs that will be fused together
        
        # Pre-compute page alignment for reuse
        alignment_map = align_pages(pages_a, pages_b, use_similarity=True)
        
        # =================================================================
        # Phase 2: Early termination for identical pages (Step 4)
        # Skip comparison for pages with identical content checksums
        # =================================================================
        page_b_lookup = {page.page_num: page for page in pages_b}
        identical_pages: set[int] = set()
        
        for page_a in pages_a:
            if page_a.page_num not in alignment_map:
                continue
            page_b_num, _ = alignment_map[page_a.page_num]
            if page_b_num not in page_b_lookup:
                continue
            
            page_b = page_b_lookup[page_b_num]
            is_ocr = page_a.page_num in ocr_pages_a or page_b_num in ocr_pages_b
            
            if _pages_are_identical(page_a, page_b, is_ocr=is_ocr):
                identical_pages.add(page_a.page_num)
                self.quality_metrics.pages_skipped_identical += 1
        
        if identical_pages:
            logger.info(
                "Phase 2 early termination: %d/%d pages are identical, skipping",
                len(identical_pages), len(pages_a)
            )
        
        # Filter out identical pages from comparison
        pages_a_filtered = [p for p in pages_a if p.page_num not in identical_pages]
        pages_b_filtered_nums = {
            alignment_map[p.page_num][0] 
            for p in pages_a_filtered 
            if p.page_num in alignment_map
        }
        pages_b_filtered = [p for p in pages_b if p.page_num in pages_b_filtered_nums]

        # Line-level comparison (primary for text diffs)
        # Avoid double-reporting text inside tables/figures: specialized comparators
        # already emit diffs anchored to those regions.
        pages_a_for_lines = _filter_lines_overlapping_regions(
            pages_a_filtered, region_key="tables", overlap_ratio=0.5
        )
        pages_b_for_lines = _filter_lines_overlapping_regions(
            pages_b_filtered, region_key="tables", overlap_ratio=0.5
        )
        pages_a_for_lines = _filter_lines_overlapping_regions(
            pages_a_for_lines, region_key="figures", overlap_ratio=0.5
        )
        pages_b_for_lines = _filter_lines_overlapping_regions(
            pages_b_for_lines, region_key="figures", overlap_ratio=0.5
        )
        # For OCR pages, line_comparison already uses paragraph merging via settings
        with track_time("line_comparison"):
            line_diffs = compare_lines(
                pages_a_for_lines, pages_b_for_lines, alignment_map=alignment_map
            )

        # Fallback: block-level comparison for pages without line data
        # (excluding identical pages from Phase 2 early termination)
        fallback_page_nums = set()
        lines_a = {p.page_num for p in pages_a_filtered if p.lines}
        lines_b = {p.page_num for p in pages_b_filtered if p.lines}
        for page_a in pages_a_filtered:
            if page_a.page_num not in alignment_map:
                continue
            page_b_num, _ = alignment_map[page_a.page_num]
            if page_a.page_num not in lines_a or page_b_num not in lines_b:
                fallback_page_nums.add(page_a.page_num)

        fallback_pages_a = [p for p in pages_a_filtered if p.page_num in fallback_page_nums]
        fallback_pages_b_nums = {alignment_map[p][0] for p in fallback_page_nums if p in alignment_map}
        fallback_pages_b = [p for p in pages_b_filtered if p.page_num in fallback_pages_b_nums]

        # Text comparison
        with track_time("text_comparison"):
            text_diffs = self._text_comparator.compare(
                fallback_pages_a,
                fallback_pages_b,
                alignment_map=alignment_map,
            )
        
        # =================================================================
        # Formatting comparison: SKIP for OCR pages (font/size is synthetic)
        # OCR engines assign placeholder font sizes, so formatting diffs
        # would be entirely noise.
        # =================================================================
        formatting_diffs: List[Diff] = []
        with track_time("formatting_comparison"):
            if any_ocr and settings.skip_formatting_for_ocr:
                logger.info(
                    "Skipping formatting comparison for OCR documents "
                    "(skip_formatting_for_ocr=True)"
                )
            else:
                formatting_diffs = compare_formatting(pages_a_filtered, pages_b_filtered)
        
        # Table comparison (use filtered pages)
        with track_time("table_comparison"):
            table_diffs = compare_tables(pages_a_filtered, pages_b_filtered)
        
        # Header/footer comparison (skip for OCR if configured)
        hf_diffs: List[Diff] = []
        with track_time("header_footer_comparison"):
            if any_ocr and settings.skip_header_footer_for_ocr:
                logger.debug("Skipping header/footer comparison for OCR pages (skip_header_footer_for_ocr=True)")
            else:
                hf_diffs = compare_headers_footers(pages_a_filtered, pages_b_filtered)
        
        # Figure caption comparison (use filtered pages)
        with track_time("figure_comparison"):
            figure_diffs = compare_figure_captions(pages_a_filtered, pages_b_filtered)
        
        self.metrics.comparison_time = time.time() - compare_start
        
        # Prepare diff lists with module names for fusion
        diff_lists = [
            ("line", line_diffs),
            ("text", text_diffs),
            ("formatting", formatting_diffs),
            ("table", table_diffs),
            ("header_footer", hf_diffs),
            ("figure", figure_diffs),
        ]
        
        # Calculate total diffs before fusion
        total_before = sum(len(diffs) for _, diffs in diff_lists)
        self.metrics.diffs_before_fusion = total_before
        
        # Apply fusion to deduplicate and triangulate confidence
        fusion_start = time.time()
        if self.config.use_fusion and total_before > 0:
            with track_time("diff_fusion"):
                all_diffs = fuse_diffs(
                    diff_lists,
                    strategy=self.config.fusion_strategy,
                    iou_threshold=self.config.fusion_iou_threshold,
                )
            logger.info("Fusion: %d diffs -> %d diffs (%.1f%% reduction)",
                       total_before, len(all_diffs),
                       (1 - len(all_diffs) / max(1, total_before)) * 100)
        else:
            # No fusion - just concatenate all diffs
            all_diffs = []
            for _, diffs in diff_lists:
                all_diffs.extend(diffs)
        
        self.metrics.fusion_time = time.time() - fusion_start
        self.metrics.diffs_after_fusion = len(all_diffs)
        
        # Phase 2: Collect timing breakdown from track_time context managers
        from utils.performance import get_timings, clear_timings
        for timing in get_timings():
            self.metrics.timing_breakdown[timing.name] = timing.duration
        clear_timings()  # Reset for next comparison

        # Tag diffs with OCR context for downstream classification/normalization.
        # This keeps OCR-only text normalization from affecting digital PDFs,
        # while still letting the classifier treat OCR dash/spacing artifacts as formatting.
        # Use the pre-computed ocr_pages_a set (already computed at the start of diff())
        for d in all_diffs:
            if d.metadata is None:
                d.metadata = {}
            d.metadata.setdefault("is_ocr", d.page_num in ocr_pages_a or any_ocr)
        
        # Classify all diffs
        classify_start = time.time()
        classified_diffs = classify_diffs(all_diffs)
        self.metrics.classification_time = time.time() - classify_start
        
        # Phase 2: Calculate quality metrics (pass OCR flag for enhanced metrics)
        self._calculate_quality_metrics(classified_diffs, is_ocr=any_ocr)
        
        # Build result
        result = ComparisonResult(
            doc1=doc1_path,
            doc2=doc2_path,
            pages=pages_a,  # Store pages from doc1
            diffs=classified_diffs,
            summary=get_diff_summary(classified_diffs),
        )
        
        logger.info("Found %d diffs in %.2fs", 
                   len(classified_diffs), 
                   self.metrics.comparison_time)
        
        return result
    
    def _calculate_quality_metrics(self, diffs: List[Diff], is_ocr: bool = False) -> None:
        """Calculate Phase 2 quality metrics from classified diffs.
        
        When OCR is used, also computes enhanced OCRQualityMetrics with:
        - Precision proxy (estimate of real vs phantom diffs)
        - Severity breakdown (critical/high/medium/low/none)
        - Phantom diff detection (whitespace, diacritics, OCR noise)
        """
        # Initialize enhanced OCR quality metrics if available and OCR was used
        ocr_quality = None
        if HAS_OCR_QUALITY_METRICS and is_ocr:
            ocr_quality = OCRQualityMetrics()
            ocr_quality.engine_used = self.config.ocr_engine or ""
            
            # Copy gating stats if available
            ocr_quality.pages_skipped_identical = self.quality_metrics.pages_skipped_identical
            ocr_quality.blocks_skipped_identical = self.quality_metrics.blocks_skipped_identical
        
        for d in diffs:
            # Count by change type
            if d.change_type == "content":
                self.quality_metrics.content_diffs += 1
            elif d.change_type == "formatting":
                self.quality_metrics.formatting_diffs += 1
            elif d.change_type == "layout":
                self.quality_metrics.layout_diffs += 1
            elif d.change_type == "visual":
                self.quality_metrics.visual_diffs += 1
            
            # Detect whitespace-only diffs (using OCR normalizer if available)
            if d.old_text and d.new_text:
                if HAS_OCR_NORMALIZER and is_ocr:
                    old_norm = normalize_ocr_compare(d.old_text)
                    new_norm = normalize_ocr_compare(d.new_text)
                else:
                    old_norm = normalize_text_full(d.old_text)
                    new_norm = normalize_text_full(d.new_text)
                
                if old_norm == new_norm:
                    self.quality_metrics.whitespace_only_diffs += 1
            
            # Detect formatting-only diffs (no semantic content change)
            if d.change_type == "formatting":
                self.quality_metrics.formatting_only_diffs += 1
            
            # Detect low-confidence diffs
            if d.confidence is not None and d.confidence < 0.5:
                self.quality_metrics.low_confidence_diffs += 1
        
        # Analyze diffs with enhanced OCR quality metrics
        if ocr_quality is not None:
            ocr_quality.analyze_diffs(diffs, is_ocr=is_ocr)
            
            # Store OCR quality metrics in pipeline metrics
            self.metrics.quality_metrics.update({
                "ocr_quality": ocr_quality.to_dict(),
                "precision_proxy": ocr_quality.precision_proxy,
                "phantom_diff_ratio": ocr_quality.phantom_diff_ratio,
                "quality_score": ocr_quality.quality_score,
                "severity_breakdown": ocr_quality.severity.to_dict(),
            })
            
            logger.info(
                "OCR Quality: precision_proxy=%.1f%%, phantom_ratio=%.1f%%, quality_score=%.0f/100",
                ocr_quality.precision_proxy * 100,
                ocr_quality.phantom_diff_ratio * 100,
                ocr_quality.quality_score,
            )
            logger.info(
                "Severity: critical=%d, high=%d, medium=%d, low=%d, phantom=%d",
                ocr_quality.severity.critical,
                ocr_quality.severity.high,
                ocr_quality.severity.medium,
                ocr_quality.severity.low,
                ocr_quality.severity.none,
            )
        
        logger.debug(
            "Quality metrics: content=%d, formatting=%d, whitespace_only=%d, phantom_proxy=%.3f",
            self.quality_metrics.content_diffs,
            self.quality_metrics.formatting_diffs,
            self.quality_metrics.whitespace_only_diffs,
            self.quality_metrics.phantom_diff_proxy,
        )


def compare_pdfs(
    pdf_a: str | Path,
    pdf_b: str | Path,
    *,
    ocr_mode: Literal["auto", "hybrid", "ocr_only"] = "auto",
    ocr_engine: Optional[str] = None,
    force_ocr: bool = False,
    sensitivity: Optional[float] = None,
    debug_mode: bool = False,
    debug_output_path: Optional[str] = None,
) -> ComparisonResult:
    """
    Compare two PDF documents end-to-end.
    
    This is the main entrypoint for programmatic usage.
    
    Args:
        pdf_a: Path to first PDF
        pdf_b: Path to second PDF
        ocr_mode: "auto" (default), "hybrid", or "ocr_only"
        ocr_engine: OCR engine to use ("deepseek", "paddle", "tesseract")
        force_ocr: Force OCR even for digital PDFs
        sensitivity: Text similarity threshold (0.0-1.0)
        debug_mode: Enable debug JSON export
        debug_output_path: Custom path for debug output
        
    Returns:
        ComparisonResult with diffs, pages, and summary
        
    Example:
        from pipeline import compare_pdfs
        
        result = compare_pdfs("doc_v1.pdf", "doc_v2.pdf")
        print(f"Found {len(result.diffs)} differences")
        
        for diff in result.diffs:
            print(f"Page {diff.page_num}: {diff.change_type} - {diff.diff_type}")
        
        # With debug output:
        result = compare_pdfs("a.pdf", "b.pdf", debug_mode=True)
    """
    config = PipelineConfig(
        ocr_mode=ocr_mode,
        ocr_engine=ocr_engine,
        force_ocr=force_ocr,
        sensitivity_threshold=sensitivity,
        debug_mode=debug_mode,
        debug_output_path=debug_output_path,
    )
    
    pipeline = ComparisonPipeline(config)
    return pipeline.compare(pdf_a, pdf_b)


def extract_single_pdf(
    pdf_path: str | Path,
    *,
    force_ocr: bool = False,
    run_layout: bool = True,
) -> List[PageData]:
    """
    Extract pages from a single PDF.
    
    Convenience function for single-document extraction with layout analysis.
    
    Args:
        pdf_path: Path to PDF file
        force_ocr: Force OCR even for digital PDFs
        run_layout: Run layout analysis to detect tables/figures
        
    Returns:
        List of PageData with text blocks and layout metadata
    """
    config = PipelineConfig(
        force_ocr=force_ocr,
        run_layout_analysis=run_layout,
    )
    
    pipeline = ComparisonPipeline(config)
    return pipeline.extract(pdf_path)
