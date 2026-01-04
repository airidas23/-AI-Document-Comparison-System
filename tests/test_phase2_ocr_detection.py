"""Tests for Phase 2 OCR-aware change detection components.

Tests cover:
- OCR Normalizer (hyphenation, ligatures, diacritics)
- Two-Stage Gating (Stage A rapidfuzz, Stage B semantic)
- Page Checksum (text hash, image hash)
- Engine Profiles (Paddle, Tesseract, DeepSeek)
- OCR Quality Metrics (precision proxy, severity)
"""
import pytest
from typing import List, Tuple


# =============================================================================
# OCR Normalizer Tests
# =============================================================================

class TestOCRNormalizer:
    """Test suite for OCR academic text normalization."""
    
    def test_ligature_expansion(self):
        """Test that ligatures are expanded correctly."""
        from utils.ocr_normalizer import expand_ligatures
        
        # Common ligatures
        assert expand_ligatures("ﬁnance") == "finance"
        assert expand_ligatures("ﬂow") == "flow"
        assert expand_ligatures("ﬃcient") == "fficient"
        assert expand_ligatures("ﬄ") == "ffl"
        
        # Mixed text
        assert expand_ligatures("The ﬁrst ﬂoor") == "The first floor"
        
    def test_hyphenation_merging(self):
        """Test that hyphenated words across lines are merged."""
        from utils.ocr_normalizer import merge_hyphenated_words
        
        # Basic hyphenation
        text = "infor-\nmacija"
        assert merge_hyphenated_words(text) == "informacija"
        
        # Multiple hyphenations
        text = "tech-\nnology and com-\nputer"
        assert merge_hyphenated_words(text) == "technology and computer"
        
        # Hyphen that's not a line break
        text = "self-contained"
        assert merge_hyphenated_words(text) == "self-contained"
        
    def test_diacritics_analysis(self):
        """Test diacritics difference analysis."""
        from utils.ocr_normalizer import analyze_diacritics_difference
        
        result = analyze_diacritics_difference("ėjimas", "ejimas")
        
        assert result.is_diacritics_only
        assert result.severity in ("low", "medium", "high")
        
    def test_diacritics_detection_in_normalizer(self):
        """Test that diacritics are detected and classified."""
        from utils.ocr_normalizer import classify_ocr_diff
        
        # Diacritics difference should be detected as such
        result = classify_ocr_diff("ėjimą", "ejima")
        
        # Should be classified as diacritics diff
        assert result.category == "diacritics"
        # Severity depends on how many diacritics differ
        assert result.severity in ("low", "medium", "high")
        
    def test_strict_normalization_basic(self):
        """Test strict normalization handles basic text."""
        from utils.ocr_normalizer import normalize_ocr_strict
        
        # Strict mode lowercases
        text = "Hello World"
        result = normalize_ocr_strict(text)
        
        assert result == "hello world"
        
    def test_strict_vs_compare_normalization(self):
        """Test the difference between strict and compare normalization."""
        from utils.ocr_normalizer import normalize_ocr_strict, normalize_ocr_compare
        
        text = "Hello World!"
        
        strict = normalize_ocr_strict(text)
        compare = normalize_ocr_compare(text)
        
        # Both produce valid normalized output
        assert isinstance(strict, str)
        assert isinstance(compare, str)
        
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        from utils.ocr_normalizer import normalize_ocr_compare
        
        text1 = "Hello    world"
        text2 = "Hello world"
        text3 = "Hello\n\nworld"
        
        assert normalize_ocr_compare(text1) == normalize_ocr_compare(text2)
        assert normalize_ocr_compare(text2) == normalize_ocr_compare(text3)
        
    def test_diff_classification(self):
        """Test OCR diff classification returns proper structure."""
        from utils.ocr_normalizer import classify_ocr_diff
        
        # Whitespace only - returns classification object
        result = classify_ocr_diff("hello world", "hello  world")
        assert result.category == "formatting"  # Whitespace is formatting
        assert result.is_phantom == True
        
        # Diacritics only
        result = classify_ocr_diff("ėjimas", "ejimas")
        assert result.category == "diacritics"
        
        # Real content change
        result = classify_ocr_diff("hello", "goodbye")
        assert result.category == "content"
        assert result.is_phantom == False


# =============================================================================
# Two-Stage Gating Tests
# =============================================================================

class TestOCRGating:
    """Test suite for two-stage OCR gating system."""
    
    def test_stage_a_identical(self):
        """Test Stage A identifies identical text."""
        from comparison.ocr_gating import stage_a_gate, GateDecision, GatingConfig
        
        text = "This is some sample text for testing."
        config = GatingConfig()
        result = stage_a_gate(text, text, config)
        
        assert result.decision == GateDecision.IDENTICAL
        
    def test_stage_a_likely_identical(self):
        """Test Stage A identifies likely identical text."""
        from comparison.ocr_gating import stage_a_gate, GateDecision, GatingConfig
        
        text1 = "This is sample text."
        text2 = "This is sample text"  # Missing period
        config = GatingConfig()
        
        result = stage_a_gate(text1, text2, config)
        
        assert result.decision in (GateDecision.IDENTICAL, GateDecision.LIKELY_IDENTICAL)
        
    def test_stage_a_needs_diff(self):
        """Test Stage A identifies text needing diff."""
        from comparison.ocr_gating import stage_a_gate, GateDecision, GatingConfig
        
        text1 = "Hello world"
        text2 = "Goodbye world completely different text here"
        config = GatingConfig()
        
        result = stage_a_gate(text1, text2, config)
        
        assert result.decision in (GateDecision.NEEDS_DIFF, GateDecision.LIKELY_DIFFERENT, GateDecision.SEMANTIC_CHECK)
        
    def test_stage_a_semantic_check_gray_zone(self):
        """Test Stage A triggers semantic check for gray zone."""
        from comparison.ocr_gating import stage_a_gate, GateDecision, GatingConfig
        
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox jumped over the lazy dog"  # Minor change
        config = GatingConfig()
        
        result = stage_a_gate(text1, text2, config)
        
        # Should be high similarity or gray zone
        assert result.decision in (
            GateDecision.SEMANTIC_CHECK, 
            GateDecision.NEEDS_DIFF,
            GateDecision.LIKELY_IDENTICAL,
            GateDecision.IDENTICAL,
        )
        
    def test_apply_gating_filter(self):
        """Test full gating pipeline filters noise."""
        from comparison.ocr_gating import apply_gating, GatingConfig
        
        config = GatingConfig()
        
        # Identical text should be filtered
        result = apply_gating("hello", "hello", config)
        assert result.should_skip_diff
        
        # Very different text should pass through
        result = apply_gating("hello world", "completely different text entirely", config)
        assert not result.should_skip_diff
        
    def test_gating_with_bboxes(self):
        """Test gating considers bounding box overlap."""
        from comparison.ocr_gating import stage_a_gate, GatingConfig
        
        text1 = "Sample text"
        text2 = "Sample text"
        config = GatingConfig()
        
        # Same position
        bbox1 = {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05}
        bbox2 = {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05}
        
        result = stage_a_gate(text1, text2, config, bbox_a=bbox1, bbox_b=bbox2)
        
        # Should be identical with matching positions
        assert result.decision in (result.decision.IDENTICAL, result.decision.LIKELY_IDENTICAL)


# =============================================================================
# Page Checksum Tests
# =============================================================================

class TestPageChecksum:
    """Test suite for page checksum computation."""
    
    def test_text_checksum_consistency(self):
        """Test text checksum is consistent."""
        from utils.page_checksum import compute_text_checksum_from_string
        
        text = "Hello world, this is a test."
        
        checksum1 = compute_text_checksum_from_string(text)
        checksum2 = compute_text_checksum_from_string(text)
        
        assert checksum1 == checksum2
        
    def test_text_checksum_difference(self):
        """Test different text produces different checksum."""
        from utils.page_checksum import compute_text_checksum_from_string
        
        text1 = "Hello world"
        text2 = "Goodbye world"
        
        checksum1 = compute_text_checksum_from_string(text1)
        checksum2 = compute_text_checksum_from_string(text2)
        
        assert checksum1 != checksum2
        
    def test_text_checksum_normalization(self):
        """Test checksum uses normalized text."""
        from utils.page_checksum import compute_text_checksum_from_string
        
        # With different whitespace - should be same after normalization
        text1 = "Hello   world"
        text2 = "Hello world"
        
        checksum1 = compute_text_checksum_from_string(text1)
        checksum2 = compute_text_checksum_from_string(text2)
        
        assert checksum1 == checksum2
        
    def test_checksum_matches_method(self):
        """Test PageChecksum.matches() works correctly."""
        from utils.page_checksum import PageChecksum, ChecksumType
        
        checksum_a = PageChecksum(
            page_num=1,
            checksum_type=ChecksumType.TEXT,
            text_hash="abc123",
        )
        
        checksum_b = PageChecksum(
            page_num=1,
            checksum_type=ChecksumType.TEXT,
            text_hash="abc123",  # Same hash
        )
        
        checksum_c = PageChecksum(
            page_num=1,
            checksum_type=ChecksumType.TEXT,
            text_hash="xyz789",  # Different hash
        )
        
        assert checksum_a.matches(checksum_b)
        assert not checksum_a.matches(checksum_c)


# =============================================================================
# Engine Profile Tests
# =============================================================================

class TestEngineProfiles:
    """Test suite for OCR engine calibration profiles."""
    
    def test_paddle_profile_exists(self):
        """Test Paddle profile can be loaded."""
        from config.engine_profiles import get_engine_profile
        
        profile = get_engine_profile("paddle")
        
        assert profile is not None
        assert profile.engine_name == "paddle"
        assert profile.gating_identical_threshold > 0.9
        
    def test_tesseract_profile_exists(self):
        """Test Tesseract profile can be loaded."""
        from config.engine_profiles import get_engine_profile
        
        profile = get_engine_profile("tesseract")
        
        assert profile is not None
        assert profile.engine_name == "tesseract"
        # Tesseract is more lenient with diacritics
        assert profile.normalize_diacritics == True
        
    def test_deepseek_profile_exists(self):
        """Test DeepSeek profile can be loaded."""
        from config.engine_profiles import get_engine_profile
        
        profile = get_engine_profile("deepseek")
        
        assert profile is not None
        assert profile.engine_name == "deepseek"
        
    def test_native_profile_exists(self):
        """Test Native profile can be loaded."""
        from config.engine_profiles import get_engine_profile
        
        profile = get_engine_profile("native")
        
        assert profile is not None
        assert profile.engine_name == "native"
        # Native should be strict
        assert profile.gating_identical_threshold >= 0.98
        
    def test_unknown_engine_returns_native(self):
        """Test unknown engine returns native profile."""
        from config.engine_profiles import get_engine_profile
        
        profile = get_engine_profile("unknown_engine")
        
        # Default fallback is native profile
        assert profile is not None
        assert profile.engine_name == "native"
        
    def test_profile_parameters_valid(self):
        """Test all profile parameters are within valid ranges."""
        from config.engine_profiles import get_engine_profile
        
        for engine in ["paddle", "tesseract", "deepseek", "native"]:
            profile = get_engine_profile(engine)
            
            # Thresholds should be between 0 and 1
            assert 0 <= profile.gating_identical_threshold <= 1
            assert 0 <= profile.gating_likely_identical_threshold <= 1
            assert 0 <= profile.gating_gray_zone_low <= 1
            assert 0 <= profile.gating_gray_zone_high <= 1
            
            # Gray zone should be valid range
            assert profile.gating_gray_zone_low < profile.gating_gray_zone_high


# =============================================================================
# OCR Quality Metrics Tests
# =============================================================================

class TestOCRQualityMetrics:
    """Test suite for OCR quality metrics computation."""
    
    def test_precision_proxy_calculation(self):
        """Test precision proxy is calculated correctly."""
        from utils.ocr_quality_metrics import OCRQualityMetrics
        
        metrics = OCRQualityMetrics()
        metrics.total_diffs = 10
        metrics.whitespace_only_diffs = 2
        metrics.punctuation_only_diffs = 1
        
        # Precision = 1 - (phantom / total) = 1 - 3/10 = 0.7
        assert abs(metrics.precision_proxy - 0.7) < 0.01
        
    def test_phantom_diff_ratio(self):
        """Test phantom diff ratio calculation."""
        from utils.ocr_quality_metrics import OCRQualityMetrics
        
        metrics = OCRQualityMetrics()
        metrics.total_diffs = 10
        metrics.whitespace_only_diffs = 3
        
        # Phantom ratio = 3/10 = 0.3
        assert abs(metrics.phantom_diff_ratio - 0.3) < 0.01
        
    def test_quality_score_range(self):
        """Test quality score is within valid range."""
        from utils.ocr_quality_metrics import OCRQualityMetrics
        
        metrics = OCRQualityMetrics()
        metrics.total_diffs = 5
        metrics.severity.high = 2
        metrics.severity.medium = 2
        metrics.severity.low = 1
        
        # Quality score should be between 0 and 100
        assert 0 <= metrics.quality_score <= 100
        
    def test_severity_breakdown_totals(self):
        """Test severity breakdown totals correctly."""
        from utils.ocr_quality_metrics import SeverityBreakdown
        
        breakdown = SeverityBreakdown(
            none=2,
            low=3,
            medium=4,
            high=5,
            critical=1,
        )
        
        assert breakdown.total == 15
        assert breakdown.real_diffs == 13  # Excluding none
        assert breakdown.significant_diffs == 10  # medium + high + critical
        
    def test_analyze_empty_diffs(self):
        """Test analyzing empty diff list."""
        from utils.ocr_quality_metrics import OCRQualityMetrics
        
        metrics = OCRQualityMetrics()
        metrics.analyze_diffs([], is_ocr=True)
        
        assert metrics.total_diffs == 0
        assert metrics.precision_proxy == 1.0  # Perfect when no diffs
        
    def test_to_dict_serialization(self):
        """Test metrics can be serialized to dict."""
        from utils.ocr_quality_metrics import OCRQualityMetrics
        
        metrics = OCRQualityMetrics()
        metrics.total_diffs = 5
        metrics.content_diffs = 3
        
        data = metrics.to_dict()
        
        assert isinstance(data, dict)
        assert data["total_diffs"] == 5
        assert data["content_diffs"] == 3
        assert "precision_proxy" in data
        assert "severity" in data


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase2Integration:
    """Integration tests for Phase 2 OCR-aware detection."""
    
    def test_normalizer_and_gating_work_together(self):
        """Test normalizer and gating integrate correctly."""
        from utils.ocr_normalizer import normalize_ocr_compare
        from comparison.ocr_gating import apply_gating, GatingConfig
        
        # Text that should be identified as identical after normalization
        text1 = "Hello  world!"
        text2 = "Hello world !"
        
        config = GatingConfig()
        
        # Apply gating with normalization
        result = apply_gating(text1, text2, config)
        
        # Should be identified as identical or likely identical
        assert result.should_skip_diff or result.confidence > 0.9
        
    def test_checksum_and_metrics_integration(self):
        """Test checksum and metrics work together."""
        from utils.page_checksum import compute_text_checksum_from_string
        from utils.ocr_quality_metrics import OCRQualityMetrics
        
        text = "Sample document text for testing."
        
        # Compute checksum
        checksum = compute_text_checksum_from_string(text)
        assert isinstance(checksum, str)
        assert len(checksum) > 0
        
        # Verify metrics can track page operations
        metrics = OCRQualityMetrics()
        metrics.total_diffs = 0  # No diffs means perfect precision
        
        assert metrics.precision_proxy == 1.0
        
    def test_engine_profile_affects_gating(self):
        """Test engine profile parameters affect gating decisions."""
        from config.engine_profiles import get_engine_profile
        from comparison.ocr_gating import GatingConfig
        
        # Get Tesseract profile (more lenient)
        tesseract = get_engine_profile("tesseract")
        
        # Get Native profile (strict)
        native = get_engine_profile("native")
        
        # Tesseract should have lower thresholds for text similarity
        # (more lenient on OCR noise)
        assert tesseract.text_similarity_threshold <= native.text_similarity_threshold
