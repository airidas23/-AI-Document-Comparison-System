"""Enhanced OCR Quality Metrics for Phase 2.

Provides comprehensive metrics for OCR comparison quality:
- Precision proxy (estimated from phantom diff detection)
- Phantom diff counter with severity breakdown
- Diffs by severity (high/medium/low)
- Engine parity metrics
- Confidence distribution analysis
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from comparison.models import Diff


class DiffSeverity(Enum):
    """Severity level for a detected difference."""
    NONE = "none"        # Not a real diff (phantom)
    LOW = "low"          # Minor/cosmetic change
    MEDIUM = "medium"    # Moderate change
    HIGH = "high"        # Significant content change
    CRITICAL = "critical"  # Major structural change


@dataclass
class SeverityBreakdown:
    """Breakdown of diffs by severity level."""
    
    none: int = 0      # Phantom diffs (should be filtered)
    low: int = 0       # Minor changes (whitespace, punctuation)
    medium: int = 0    # Moderate changes (formatting, minor text)
    high: int = 0      # Significant changes (content, structure)
    critical: int = 0  # Major changes (deleted sections, etc.)
    
    @property
    def total(self) -> int:
        return self.none + self.low + self.medium + self.high + self.critical
    
    @property
    def real_diffs(self) -> int:
        """Diffs that are actually meaningful (excluding phantom)."""
        return self.low + self.medium + self.high + self.critical
    
    @property
    def significant_diffs(self) -> int:
        """Diffs that require attention."""
        return self.medium + self.high + self.critical
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "none": self.none,
            "low": self.low,
            "medium": self.medium,
            "high": self.high,
            "critical": self.critical,
            "total": self.total,
            "real_diffs": self.real_diffs,
            "significant_diffs": self.significant_diffs,
        }


@dataclass
class OCRQualityMetrics:
    """Comprehensive OCR quality metrics.
    
    Tracks phantom diffs, severity distribution, and estimated precision.
    """
    
    # Basic counts
    total_diffs: int = 0
    content_diffs: int = 0
    formatting_diffs: int = 0
    layout_diffs: int = 0
    visual_diffs: int = 0
    
    # Phantom diff indicators
    phantom_diffs_detected: int = 0
    whitespace_only_diffs: int = 0
    punctuation_only_diffs: int = 0
    case_only_diffs: int = 0
    diacritics_only_diffs: int = 0
    ocr_noise_diffs: int = 0
    
    # Severity breakdown
    severity: SeverityBreakdown = field(default_factory=SeverityBreakdown)
    
    # Confidence metrics
    low_confidence_diffs: int = 0   # confidence < 0.5
    high_confidence_diffs: int = 0  # confidence > 0.85
    avg_diff_confidence: float = 0.0
    
    # Gating statistics (from two-stage gating)
    pairs_checked: int = 0
    pairs_skipped_by_gating: int = 0
    pairs_semantic_checked: int = 0
    
    # Alignment quality
    unmatched_blocks_a: int = 0
    unmatched_blocks_b: int = 0
    
    # Early termination stats
    pages_skipped_identical: int = 0
    blocks_skipped_identical: int = 0
    
    # Engine info
    engine_used: str = ""
    engine_profile_applied: bool = False
    
    @property
    def phantom_diff_total(self) -> int:
        """Total count of likely phantom diffs."""
        return (
            self.whitespace_only_diffs +
            self.punctuation_only_diffs +
            self.case_only_diffs +
            self.diacritics_only_diffs +
            self.ocr_noise_diffs
        )
    
    @property
    def precision_proxy(self) -> float:
        """Estimated precision (real diffs / total diffs).
        
        Higher is better. Target: 0.8+
        """
        if self.total_diffs == 0:
            return 1.0
        return 1.0 - (self.phantom_diff_total / self.total_diffs)
    
    @property
    def phantom_diff_ratio(self) -> float:
        """Ratio of phantom diffs (should decrease with optimization).
        
        Lower is better. Target: <0.2
        """
        if self.total_diffs == 0:
            return 0.0
        return self.phantom_diff_total / self.total_diffs
    
    @property
    def whitespace_only_ratio(self) -> float:
        """Ratio of whitespace-only diffs."""
        if self.total_diffs == 0:
            return 0.0
        return self.whitespace_only_diffs / self.total_diffs
    
    @property
    def gating_skip_rate(self) -> float:
        """Rate of pairs skipped by gating (higher = more efficient)."""
        if self.pairs_checked == 0:
            return 0.0
        return self.pairs_skipped_by_gating / self.pairs_checked
    
    @property
    def quality_score(self) -> float:
        """Overall quality score (0-100).
        
        Composite metric considering precision, phantom ratio, and severity.
        """
        # Precision component (40%)
        precision_score = self.precision_proxy * 40
        
        # Phantom ratio component (30%) - inverted
        phantom_score = (1 - self.phantom_diff_ratio) * 30
        
        # Severity distribution component (30%)
        if self.severity.total > 0:
            # Prefer high/critical diffs over low/none
            severity_score = (
                self.severity.significant_diffs / self.severity.total
            ) * 30
        else:
            severity_score = 30  # No diffs = perfect
        
        return precision_score + phantom_score + severity_score
    
    def classify_diff_severity(
        self,
        diff: "Diff",
        is_ocr: bool = True,
    ) -> DiffSeverity:
        """Classify the severity of a single diff.
        
        Args:
            diff: The diff to classify
            is_ocr: Whether OCR was used (affects classification)
            
        Returns:
            DiffSeverity level
        """
        from utils.ocr_normalizer import (
            normalize_ocr_strict,
            normalize_ocr_compare,
            analyze_diacritics_difference,
        )
        
        old_text = diff.old_text or ""
        new_text = diff.new_text or ""
        
        # Empty check
        if not old_text and not new_text:
            return DiffSeverity.NONE
        
        # Added/deleted = high severity
        if not old_text or not new_text:
            return DiffSeverity.HIGH
        
        # Check if identical after normalization
        norm_old = normalize_ocr_compare(old_text)
        norm_new = normalize_ocr_compare(new_text)
        
        if norm_old == norm_new:
            # Whitespace/formatting only
            self.whitespace_only_diffs += 1
            return DiffSeverity.NONE if is_ocr else DiffSeverity.LOW
        
        # Check strict normalization
        strict_old = normalize_ocr_strict(old_text)
        strict_new = normalize_ocr_strict(new_text)
        
        if strict_old == strict_new:
            # Case/punctuation only
            self.punctuation_only_diffs += 1
            return DiffSeverity.NONE if is_ocr else DiffSeverity.LOW
        
        # Check diacritics difference
        diacritics = analyze_diacritics_difference(old_text, new_text)
        if diacritics.is_diacritics_only:
            self.diacritics_only_diffs += 1
            if diacritics.severity == "low":
                return DiffSeverity.NONE if is_ocr else DiffSeverity.LOW
            elif diacritics.severity == "medium":
                return DiffSeverity.LOW
            else:
                return DiffSeverity.MEDIUM
        
        # Confidence-based classification
        confidence = diff.confidence or 0.5
        if confidence < 0.4:
            self.low_confidence_diffs += 1
            if is_ocr:
                self.ocr_noise_diffs += 1
                return DiffSeverity.LOW
        elif confidence > 0.85:
            self.high_confidence_diffs += 1
        
        # Change type based classification
        change_type = diff.change_type or "content"
        
        if change_type == "formatting":
            return DiffSeverity.LOW
        elif change_type == "layout":
            return DiffSeverity.MEDIUM
        elif change_type == "visual":
            return DiffSeverity.MEDIUM
        else:  # content
            # Check text similarity for severity
            from rapidfuzz.fuzz import ratio
            similarity = ratio(norm_old, norm_new) / 100.0
            
            if similarity > 0.9:
                return DiffSeverity.LOW
            elif similarity > 0.7:
                return DiffSeverity.MEDIUM
            elif similarity > 0.5:
                return DiffSeverity.HIGH
            else:
                return DiffSeverity.CRITICAL
    
    def analyze_diffs(
        self,
        diffs: List["Diff"],
        is_ocr: bool = True,
    ) -> None:
        """Analyze a list of diffs and populate metrics.
        
        Args:
            diffs: List of Diff objects to analyze
            is_ocr: Whether OCR was used
        """
        self.total_diffs = len(diffs)
        
        confidences = []
        
        for diff in diffs:
            # Count by type
            change_type = diff.change_type or "content"
            if change_type == "content":
                self.content_diffs += 1
            elif change_type == "formatting":
                self.formatting_diffs += 1
            elif change_type == "layout":
                self.layout_diffs += 1
            elif change_type == "visual":
                self.visual_diffs += 1
            
            # Classify severity
            severity = self.classify_diff_severity(diff, is_ocr)
            
            # Update severity counts
            if severity == DiffSeverity.NONE:
                self.severity.none += 1
                self.phantom_diffs_detected += 1
            elif severity == DiffSeverity.LOW:
                self.severity.low += 1
            elif severity == DiffSeverity.MEDIUM:
                self.severity.medium += 1
            elif severity == DiffSeverity.HIGH:
                self.severity.high += 1
            elif severity == DiffSeverity.CRITICAL:
                self.severity.critical += 1
            
            # Collect confidence
            if diff.confidence is not None:
                confidences.append(diff.confidence)
        
        # Calculate average confidence
        if confidences:
            self.avg_diff_confidence = sum(confidences) / len(confidences)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics to dictionary."""
        return {
            # Basic counts
            "total_diffs": self.total_diffs,
            "content_diffs": self.content_diffs,
            "formatting_diffs": self.formatting_diffs,
            "layout_diffs": self.layout_diffs,
            "visual_diffs": self.visual_diffs,
            
            # Phantom diff indicators
            "phantom_diffs_detected": self.phantom_diffs_detected,
            "phantom_diff_total": self.phantom_diff_total,
            "whitespace_only_diffs": self.whitespace_only_diffs,
            "punctuation_only_diffs": self.punctuation_only_diffs,
            "case_only_diffs": self.case_only_diffs,
            "diacritics_only_diffs": self.diacritics_only_diffs,
            "ocr_noise_diffs": self.ocr_noise_diffs,
            
            # Quality metrics
            "precision_proxy": round(self.precision_proxy, 4),
            "phantom_diff_ratio": round(self.phantom_diff_ratio, 4),
            "whitespace_only_ratio": round(self.whitespace_only_ratio, 4),
            "quality_score": round(self.quality_score, 2),
            
            # Severity breakdown
            "severity": self.severity.to_dict(),
            
            # Confidence metrics
            "low_confidence_diffs": self.low_confidence_diffs,
            "high_confidence_diffs": self.high_confidence_diffs,
            "avg_diff_confidence": round(self.avg_diff_confidence, 4),
            
            # Gating statistics
            "pairs_checked": self.pairs_checked,
            "pairs_skipped_by_gating": self.pairs_skipped_by_gating,
            "pairs_semantic_checked": self.pairs_semantic_checked,
            "gating_skip_rate": round(self.gating_skip_rate, 4),
            
            # Alignment quality
            "unmatched_blocks_a": self.unmatched_blocks_a,
            "unmatched_blocks_b": self.unmatched_blocks_b,
            
            # Early termination
            "pages_skipped_identical": self.pages_skipped_identical,
            "blocks_skipped_identical": self.blocks_skipped_identical,
            
            # Engine info
            "engine_used": self.engine_used,
            "engine_profile_applied": self.engine_profile_applied,
        }
    
    def get_summary_text(self) -> str:
        """Get human-readable summary of metrics."""
        lines = [
            "📊 **OCR Quality Metrics**",
            "",
            f"**Diffs Found:** {self.total_diffs}",
            f"- Content: {self.content_diffs}",
            f"- Formatting: {self.formatting_diffs}",
            f"- Layout: {self.layout_diffs}",
            "",
            "**Quality Indicators:**",
            f"- Precision Proxy: {self.precision_proxy:.1%}",
            f"- Phantom Diff Ratio: {self.phantom_diff_ratio:.1%}",
            f"- Quality Score: {self.quality_score:.0f}/100",
            "",
            "**Severity Breakdown:**",
            f"- 🔴 Critical: {self.severity.critical}",
            f"- 🟠 High: {self.severity.high}",
            f"- 🟡 Medium: {self.severity.medium}",
            f"- 🟢 Low: {self.severity.low}",
            f"- ⚪ Phantom: {self.severity.none}",
        ]
        
        if self.pairs_checked > 0:
            lines.extend([
                "",
                "**Gating Efficiency:**",
                f"- Skip Rate: {self.gating_skip_rate:.1%}",
                f"- Semantic Checks: {self.pairs_semantic_checked}",
            ])
        
        if self.pages_skipped_identical > 0:
            lines.extend([
                "",
                "**Early Termination:**",
                f"- Pages Skipped: {self.pages_skipped_identical}",
            ])
        
        return "\n".join(lines)


@dataclass  
class EngineParity:
    """Metrics comparing outputs between OCR engines."""
    
    engine_a: str = ""
    engine_b: str = ""
    
    # Text similarity
    avg_text_similarity: float = 0.0
    min_text_similarity: float = 0.0
    max_text_similarity: float = 0.0
    
    # Diff stability
    diff_count_engine_a: int = 0
    diff_count_engine_b: int = 0
    shared_diffs: int = 0
    
    # Parity score (0-1, higher = more similar)
    parity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_a": self.engine_a,
            "engine_b": self.engine_b,
            "avg_text_similarity": round(self.avg_text_similarity, 4),
            "min_text_similarity": round(self.min_text_similarity, 4),
            "max_text_similarity": round(self.max_text_similarity, 4),
            "diff_count_engine_a": self.diff_count_engine_a,
            "diff_count_engine_b": self.diff_count_engine_b,
            "shared_diffs": self.shared_diffs,
            "parity_score": round(self.parity_score, 4),
        }


def compute_engine_parity(
    text_pairs: List[tuple[str, str]],
    engine_a_name: str = "engine_a",
    engine_b_name: str = "engine_b",
) -> EngineParity:
    """Compute parity metrics between two OCR engines.
    
    Args:
        text_pairs: List of (text_a, text_b) from the two engines
        engine_a_name: Name of first engine
        engine_b_name: Name of second engine
        
    Returns:
        EngineParity with similarity metrics
    """
    from rapidfuzz.fuzz import ratio
    from utils.ocr_normalizer import normalize_ocr_compare
    
    parity = EngineParity(
        engine_a=engine_a_name,
        engine_b=engine_b_name,
    )
    
    if not text_pairs:
        return parity
    
    similarities = []
    for text_a, text_b in text_pairs:
        norm_a = normalize_ocr_compare(text_a)
        norm_b = normalize_ocr_compare(text_b)
        sim = ratio(norm_a, norm_b) / 100.0
        similarities.append(sim)
    
    parity.avg_text_similarity = sum(similarities) / len(similarities)
    parity.min_text_similarity = min(similarities)
    parity.max_text_similarity = max(similarities)
    parity.parity_score = parity.avg_text_similarity
    
    return parity
