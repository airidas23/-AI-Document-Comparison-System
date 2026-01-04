"""Per-engine calibration profiles for OCR engines.

Phase 2: Different OCR engines have different characteristics:
- Paddle: High confidence scores, fast, good for structured docs
- Tesseract: Variable confidence, works without GPU, legacy
- DeepSeek: Best quality but slow, may hallucinate

Each engine needs different threshold calibration for optimal results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class EngineProfile:
    """Calibration profile for an OCR engine."""
    
    # Engine identification
    engine_name: str
    
    # Text similarity thresholds
    text_similarity_threshold: float = 0.82
    semantic_change_threshold: float = 0.50
    
    # Gating thresholds (Phase 2 two-stage gating)
    gating_identical_threshold: float = 0.98
    gating_likely_identical_threshold: float = 0.92
    gating_gray_zone_low: float = 0.70
    gating_gray_zone_high: float = 0.85
    
    # Confidence handling
    min_confidence_threshold: float = 0.3
    use_confidence_weighting: bool = True
    confidence_weight_power: float = 1.0  # Higher = more aggressive filtering
    
    # OCR-specific noise filters
    ignore_punctuation_diffs: bool = True
    ignore_whitespace_diffs: bool = True
    ignore_case_diffs: bool = False
    aggressive_noise_filter: bool = False
    
    # OCR normalization
    merge_hyphenated_words: bool = True
    expand_ligatures: bool = True
    normalize_diacritics: bool = False  # True = treat ė≈e as same
    
    # Change significance thresholds
    min_change_chars: int = 2
    min_change_ratio: float = 0.015
    
    # Layout handling
    skip_formatting_comparison: bool = True
    skip_header_footer_comparison: bool = True
    layout_position_tolerance: float = 0.05
    
    # Table handling
    table_cell_text_threshold: float = 0.85
    table_structure_confidence_threshold: float = 0.5
    
    # Performance
    render_dpi: int = 100
    use_image_checksum: bool = False
    
    # Quality gates
    min_chars_per_page: int = 25
    min_avg_confidence: float = 0.55
    max_gibberish_ratio: float = 0.35
    
    def to_dict(self) -> Dict[str, Any]:
        """Export profile to dictionary."""
        return {
            "engine_name": self.engine_name,
            "text_similarity_threshold": self.text_similarity_threshold,
            "semantic_change_threshold": self.semantic_change_threshold,
            "gating_identical_threshold": self.gating_identical_threshold,
            "gating_likely_identical_threshold": self.gating_likely_identical_threshold,
            "gating_gray_zone_low": self.gating_gray_zone_low,
            "gating_gray_zone_high": self.gating_gray_zone_high,
            "min_confidence_threshold": self.min_confidence_threshold,
            "use_confidence_weighting": self.use_confidence_weighting,
            "confidence_weight_power": self.confidence_weight_power,
            "ignore_punctuation_diffs": self.ignore_punctuation_diffs,
            "ignore_whitespace_diffs": self.ignore_whitespace_diffs,
            "ignore_case_diffs": self.ignore_case_diffs,
            "aggressive_noise_filter": self.aggressive_noise_filter,
            "min_change_chars": self.min_change_chars,
            "min_change_ratio": self.min_change_ratio,
            "skip_formatting_comparison": self.skip_formatting_comparison,
            "skip_header_footer_comparison": self.skip_header_footer_comparison,
            "layout_position_tolerance": self.layout_position_tolerance,
            "table_cell_text_threshold": self.table_cell_text_threshold,
            "render_dpi": self.render_dpi,
            "use_image_checksum": self.use_image_checksum,
        }


# =============================================================================
# Pre-defined Engine Profiles
# =============================================================================

def _paddle_profile() -> EngineProfile:
    """PaddleOCR profile - fast, high confidence, good for structured docs."""
    return EngineProfile(
        engine_name="paddle",
        # Paddle has high confidence scores, can use tighter thresholds
        text_similarity_threshold=0.85,
        semantic_change_threshold=0.50,
        # Gating thresholds
        gating_identical_threshold=0.96,
        gating_likely_identical_threshold=0.90,
        gating_gray_zone_low=0.70,
        gating_gray_zone_high=0.85,
        # Paddle confidence is reliable - use aggressive filtering
        min_confidence_threshold=0.5,
        use_confidence_weighting=True,
        confidence_weight_power=1.5,
        # Standard OCR filters
        ignore_punctuation_diffs=True,
        ignore_whitespace_diffs=True,
        ignore_case_diffs=False,
        aggressive_noise_filter=True,
        # Normalization
        merge_hyphenated_words=True,
        expand_ligatures=True,
        normalize_diacritics=False,
        # Change thresholds
        min_change_chars=2,
        min_change_ratio=0.02,
        # Layout
        skip_formatting_comparison=True,
        skip_header_footer_comparison=True,
        layout_position_tolerance=0.04,
        # Tables
        table_cell_text_threshold=0.80,
        table_structure_confidence_threshold=0.6,
        # Performance (Paddle is fast, can use higher DPI)
        render_dpi=100,
        use_image_checksum=True,  # Use image hash for scanned early termination
        # Quality gates
        min_chars_per_page=25,
        min_avg_confidence=0.60,
        max_gibberish_ratio=0.30,
    )


def _tesseract_profile() -> EngineProfile:
    """Tesseract profile - variable quality, works without GPU."""
    return EngineProfile(
        engine_name="tesseract",
        # Tesseract is noisier - use looser thresholds
        text_similarity_threshold=0.78,
        semantic_change_threshold=0.45,
        # Gating thresholds (more lenient for noisy OCR)
        gating_identical_threshold=0.94,
        gating_likely_identical_threshold=0.86,
        gating_gray_zone_low=0.65,
        gating_gray_zone_high=0.82,
        # Tesseract confidence varies - moderate filtering
        min_confidence_threshold=0.35,
        use_confidence_weighting=True,
        confidence_weight_power=1.0,
        # OCR filters (aggressive for tesseract noise)
        ignore_punctuation_diffs=True,
        ignore_whitespace_diffs=True,
        ignore_case_diffs=True,  # Tesseract often confuses case
        aggressive_noise_filter=True,
        # Normalization
        merge_hyphenated_words=True,
        expand_ligatures=True,
        normalize_diacritics=True,  # Tesseract struggles with diacritics
        # Change thresholds (higher to filter noise)
        min_change_chars=3,
        min_change_ratio=0.025,
        # Layout
        skip_formatting_comparison=True,
        skip_header_footer_comparison=True,
        layout_position_tolerance=0.06,
        # Tables (Tesseract tables are often problematic)
        table_cell_text_threshold=0.70,
        table_structure_confidence_threshold=0.7,
        # Performance
        render_dpi=100,
        use_image_checksum=True,
        # Quality gates (more tolerant)
        min_chars_per_page=20,
        min_avg_confidence=0.45,
        max_gibberish_ratio=0.40,
    )


def _deepseek_profile() -> EngineProfile:
    """DeepSeek profile - best quality but slow, may hallucinate."""
    return EngineProfile(
        engine_name="deepseek",
        # DeepSeek is high quality - can use tight thresholds
        text_similarity_threshold=0.88,
        semantic_change_threshold=0.55,
        # Gating thresholds
        gating_identical_threshold=0.97,
        gating_likely_identical_threshold=0.92,
        gating_gray_zone_low=0.72,
        gating_gray_zone_high=0.88,
        # DeepSeek doesn't provide confidence - disable weighting
        min_confidence_threshold=0.0,
        use_confidence_weighting=False,
        confidence_weight_power=1.0,
        # Less aggressive filters (DeepSeek is accurate)
        ignore_punctuation_diffs=True,
        ignore_whitespace_diffs=True,
        ignore_case_diffs=False,
        aggressive_noise_filter=False,  # Don't over-filter good output
        # Normalization
        merge_hyphenated_words=True,
        expand_ligatures=True,
        normalize_diacritics=False,  # DeepSeek handles diacritics well
        # Change thresholds
        min_change_chars=2,
        min_change_ratio=0.015,
        # Layout
        skip_formatting_comparison=True,
        skip_header_footer_comparison=False,  # DeepSeek is good at headers
        layout_position_tolerance=0.03,
        # Tables (DeepSeek is good at tables)
        table_cell_text_threshold=0.85,
        table_structure_confidence_threshold=0.5,
        # Performance (DeepSeek is slow - use lower DPI)
        render_dpi=72,
        use_image_checksum=False,  # Already slow, skip image hash
        # Quality gates
        min_chars_per_page=30,
        min_avg_confidence=0.0,  # No confidence from DeepSeek
        max_gibberish_ratio=0.25,  # Stricter for hallucination detection
    )


def _native_profile() -> EngineProfile:
    """Native PyMuPDF profile - digital documents without OCR."""
    return EngineProfile(
        engine_name="native",
        # Native extraction is very accurate
        text_similarity_threshold=0.90,
        semantic_change_threshold=0.60,
        # Gating thresholds (strict for digital)
        gating_identical_threshold=0.99,
        gating_likely_identical_threshold=0.95,
        gating_gray_zone_low=0.80,
        gating_gray_zone_high=0.92,
        # No confidence weighting needed
        min_confidence_threshold=0.0,
        use_confidence_weighting=False,
        confidence_weight_power=1.0,
        # Minimal noise filters (native is clean)
        ignore_punctuation_diffs=False,
        ignore_whitespace_diffs=False,
        ignore_case_diffs=False,
        aggressive_noise_filter=False,
        # Normalization
        merge_hyphenated_words=False,  # Digital PDFs have accurate hyphenation
        expand_ligatures=True,
        normalize_diacritics=False,
        # Change thresholds
        min_change_chars=1,
        min_change_ratio=0.01,
        # Layout (enable for digital)
        skip_formatting_comparison=False,
        skip_header_footer_comparison=False,
        layout_position_tolerance=0.02,
        # Tables
        table_cell_text_threshold=0.90,
        table_structure_confidence_threshold=0.3,
        # Performance
        render_dpi=144,
        use_image_checksum=False,  # Text checksum is sufficient
        # Quality gates
        min_chars_per_page=10,
        min_avg_confidence=0.0,
        max_gibberish_ratio=0.10,
    )


# =============================================================================
# Engine Profile Registry
# =============================================================================

ENGINE_PROFILES: Dict[str, EngineProfile] = {
    "paddle": _paddle_profile(),
    "tesseract": _tesseract_profile(),
    "deepseek": _deepseek_profile(),
    "native": _native_profile(),
}


def get_engine_profile(engine_name: str) -> EngineProfile:
    """Get calibration profile for an OCR engine.
    
    Args:
        engine_name: Engine name ("paddle", "tesseract", "deepseek", "native")
        
    Returns:
        EngineProfile with calibrated thresholds
        
    Example:
        profile = get_engine_profile("paddle")
        if similarity > profile.text_similarity_threshold:
            # Texts are similar
    """
    engine_lower = engine_name.lower()
    
    if engine_lower in ENGINE_PROFILES:
        return ENGINE_PROFILES[engine_lower]
    
    # Default to native profile for unknown engines
    return ENGINE_PROFILES["native"]


def get_profile_for_pages(
    pages: list,
    default_engine: str = "native",
) -> EngineProfile:
    """Detect engine from pages and return appropriate profile.
    
    Args:
        pages: List of PageData
        default_engine: Engine to use if detection fails
        
    Returns:
        EngineProfile for detected or default engine
    """
    detected_engine = None
    
    for page in pages:
        md = page.metadata or {}
        
        # Check extraction method
        extraction_method = str(md.get("extraction_method", "")).lower()
        ocr_engine = str(md.get("ocr_engine_used", "")).lower()
        
        if "paddle" in ocr_engine or "paddle" in extraction_method:
            detected_engine = "paddle"
            break
        elif "tesseract" in ocr_engine or "tesseract" in extraction_method:
            detected_engine = "tesseract"
            break
        elif "deepseek" in ocr_engine or "deepseek" in extraction_method:
            detected_engine = "deepseek"
            break
        elif "ocr" in extraction_method:
            # Generic OCR, use paddle profile (most common)
            detected_engine = "paddle"
    
    if detected_engine is None:
        detected_engine = default_engine
    
    return get_engine_profile(detected_engine)


# =============================================================================
# Profile Application
# =============================================================================

def apply_profile_to_settings(profile: EngineProfile) -> None:
    """Apply engine profile thresholds to global settings.
    
    This temporarily overrides settings for the current comparison.
    
    Args:
        profile: EngineProfile to apply
    """
    from config.settings import settings
    
    # Text thresholds
    settings.text_similarity_threshold = profile.text_similarity_threshold
    settings.semantic_change_threshold = profile.semantic_change_threshold
    
    # OCR-specific settings
    settings.ocr_ignore_punctuation_diffs = profile.ignore_punctuation_diffs
    settings.ocr_ignore_whitespace_diffs = profile.ignore_whitespace_diffs
    settings.ocr_ignore_case_diffs = profile.ignore_case_diffs
    settings.ocr_aggressive_noise_filter = profile.aggressive_noise_filter
    settings.ocr_min_change_chars = profile.min_change_chars
    settings.ocr_min_change_ratio = profile.min_change_ratio
    
    # Layout settings
    settings.skip_formatting_for_ocr = profile.skip_formatting_comparison
    settings.skip_header_footer_for_ocr = profile.skip_header_footer_comparison
    settings.ocr_layout_position_tolerance = profile.layout_position_tolerance
    
    # Table settings
    settings.ocr_table_cell_text_threshold = profile.table_cell_text_threshold
    settings.ocr_table_structure_confidence_threshold = profile.table_structure_confidence_threshold


@dataclass
class EngineCalibrationResult:
    """Result of applying engine calibration."""
    
    engine_name: str
    profile: EngineProfile
    pages_analyzed: int = 0
    
    # Detection stats
    detected_from_metadata: bool = False
    confidence_stats: Dict[str, float] = field(default_factory=dict)
    
    # Applied adjustments
    threshold_adjustments: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "pages_analyzed": self.pages_analyzed,
            "detected_from_metadata": self.detected_from_metadata,
            "confidence_stats": self.confidence_stats,
            "threshold_adjustments": self.threshold_adjustments,
            "profile": self.profile.to_dict(),
        }


def calibrate_for_document(
    pages: list,
    apply_to_settings: bool = True,
) -> EngineCalibrationResult:
    """Calibrate thresholds based on document characteristics.
    
    Analyzes the document and returns/applies appropriate profile.
    
    Args:
        pages: List of PageData
        apply_to_settings: Whether to apply profile to global settings
        
    Returns:
        EngineCalibrationResult with profile and stats
    """
    profile = get_profile_for_pages(pages)
    
    result = EngineCalibrationResult(
        engine_name=profile.engine_name,
        profile=profile,
        pages_analyzed=len(pages),
        detected_from_metadata=True,
    )
    
    # Collect confidence statistics
    confidences = []
    for page in pages:
        for block in (page.blocks or []):
            if block.confidence and block.confidence > 0:
                confidences.append(block.confidence)
    
    if confidences:
        import statistics
        result.confidence_stats = {
            "mean": statistics.mean(confidences),
            "median": statistics.median(confidences),
            "stdev": statistics.stdev(confidences) if len(confidences) > 1 else 0,
            "min": min(confidences),
            "max": max(confidences),
            "count": len(confidences),
        }
        
        # Dynamic threshold adjustment based on confidence distribution
        mean_conf = result.confidence_stats["mean"]
        if mean_conf < 0.5 and profile.use_confidence_weighting:
            # Low confidence - use more lenient thresholds
            adjustment = 0.05
            result.threshold_adjustments["text_similarity_threshold"] = -adjustment
            profile.text_similarity_threshold = max(0.5, profile.text_similarity_threshold - adjustment)
    
    if apply_to_settings:
        apply_profile_to_settings(profile)
    
    return result
