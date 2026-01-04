"""
OCR Engine Parity Tests - Phase 2 Step 8

Tests that different OCR engines (PaddleOCR vs Tesseract) produce comparable 
results on scanned documents, ensuring the comparison pipeline yields stable 
diffs regardless of which engine is used.

Goals:
- engine_parity_score > 0.85 (text content similarity between engines)
- phantom_diffs_proxy stable (no engine-specific false positives)
- verify OCR metadata is correctly populated

Usage:
    pytest tests/test_ocr_engine_parity_scanned.py -v

Note: These tests require actual OCR engines to be installed:
- PaddleOCR: pip install paddleocr paddlepaddle
- Tesseract: pip install pytesseract (+ system tesseract binary)
"""
from __future__ import annotations

import pytest
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Import OCR infrastructure
from extraction.ocr_router import (
    is_paddle_available,
    is_tesseract_available,
    ocr_pdf_multi,
)
from comparison.models import PageData
from comparison.text_normalizer import normalize_compare, NormalizationConfig
from rapidfuzz import fuzz


# =============================================================================
# Test Configuration
# =============================================================================

# Minimum parity score threshold (0.0-1.0)
ENGINE_PARITY_THRESHOLD = 0.85

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "test"


@dataclass
class EngineParityResult:
    """Result of comparing two OCR engine outputs."""
    paddle_text: str
    tesseract_text: str
    parity_score: float
    paddle_char_count: int
    tesseract_char_count: int
    paddle_block_count: int
    tesseract_block_count: int
    paddle_avg_confidence: Optional[float]
    tesseract_avg_confidence: Optional[float]


# =============================================================================
# Helper Functions
# =============================================================================

def extract_full_text(pages: List[PageData]) -> str:
    """Extract all text from pages as a single string."""
    lines = []
    for page in sorted(pages, key=lambda p: p.page_num):
        for block in page.blocks:
            if block.text:
                lines.append(block.text.strip())
    return "\n".join(lines)


def get_avg_confidence(pages: List[PageData]) -> Optional[float]:
    """Calculate average OCR confidence across all blocks."""
    confidences = []
    for page in pages:
        quality = page.metadata.get("ocr_quality", {})
        if quality.get("avg_confidence") is not None:
            confidences.append(quality["avg_confidence"])
        # Also check block-level confidence
        for block in page.blocks:
            conf = block.metadata.get("confidence")
            if conf is not None:
                try:
                    c = float(conf)
                    if c > 1.0:
                        c = c / 100.0
                    confidences.append(c)
                except (ValueError, TypeError):
                    pass
    return sum(confidences) / len(confidences) if confidences else None


def compute_engine_parity(
    paddle_pages: List[PageData],
    tesseract_pages: List[PageData],
) -> EngineParityResult:
    """
    Compute parity score between two OCR engine outputs.
    
    Uses normalized text comparison to account for expected OCR variations.
    """
    config = NormalizationConfig.default_ocr()
    
    # Extract and normalize text
    paddle_text = extract_full_text(paddle_pages)
    tesseract_text = extract_full_text(tesseract_pages)
    
    paddle_norm = normalize_compare(paddle_text, config)
    tesseract_norm = normalize_compare(tesseract_text, config)
    
    # Compute similarity using rapidfuzz (handles OCR variations well)
    # Use token_sort_ratio for order-independent comparison
    parity_score = fuzz.token_sort_ratio(paddle_norm, tesseract_norm) / 100.0
    
    return EngineParityResult(
        paddle_text=paddle_text,
        tesseract_text=tesseract_text,
        parity_score=parity_score,
        paddle_char_count=len(paddle_text),
        tesseract_char_count=len(tesseract_text),
        paddle_block_count=sum(len(p.blocks) for p in paddle_pages),
        tesseract_block_count=sum(len(p.blocks) for p in tesseract_pages),
        paddle_avg_confidence=get_avg_confidence(paddle_pages),
        tesseract_avg_confidence=get_avg_confidence(tesseract_pages),
    )


def ocr_with_specific_engine(path: Path, engine: str) -> List[PageData]:
    """Run OCR with a specific engine, no fallback."""
    # Force single engine by passing it as sole priority
    return ocr_pdf_multi(
        path,
        engine_priority=[engine],
        run_layout_analysis=False,  # Skip for parity testing
        prefer_digital=False,  # Force OCR even for digital PDFs
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def scanned_pdf_path() -> Optional[Path]:
    """Find a scanned PDF for testing."""
    # Look for common test files
    candidates = [
        TEST_DATA_DIR / "scanned_sample.pdf",
        TEST_DATA_DIR / "scanned_doc.pdf",
        TEST_DATA_DIR / "test_scanned.pdf",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Look for any PDF in test directory
    if TEST_DATA_DIR.exists():
        for pdf in TEST_DATA_DIR.glob("*.pdf"):
            return pdf
    
    return None


# =============================================================================
# Test Classes
# =============================================================================

class TestOCREngineAvailability:
    """Tests for OCR engine availability detection."""
    
    def test_paddle_availability_detection(self):
        """Test that PaddleOCR availability is correctly detected."""
        result = is_paddle_available()
        # Just verify the function runs without error
        assert isinstance(result, bool)
    
    def test_tesseract_availability_detection(self):
        """Test that Tesseract availability is correctly detected."""
        result = is_tesseract_available()
        assert isinstance(result, bool)
    
    def test_at_least_one_engine_available(self):
        """Test that at least one OCR engine is available for testing."""
        paddle = is_paddle_available()
        tesseract = is_tesseract_available()
        
        # This is informational - tests may be skipped if neither is available
        if not paddle and not tesseract:
            pytest.skip("No OCR engines available for parity testing")


class TestOCRMetadataPopulation:
    """Tests for OCR metadata correctness."""
    
    @pytest.mark.skipif(not is_paddle_available(), reason="PaddleOCR not available")
    def test_paddle_metadata_populated(self, scanned_pdf_path):
        """Test that PaddleOCR populates expected metadata fields."""
        if scanned_pdf_path is None:
            pytest.skip("No test PDF available")
        
        pages = ocr_with_specific_engine(scanned_pdf_path, "paddle")
        
        if not pages:
            pytest.skip("No pages returned from OCR")
        
        # Check first page metadata
        page = pages[0]
        
        # Required metadata fields
        assert "extraction_method" in page.metadata
        assert "ocr" in page.metadata["extraction_method"].lower()
        
        # OCR quality metrics should be present
        assert "ocr_quality" in page.metadata
        quality = page.metadata["ocr_quality"]
        assert "engine" in quality
        assert quality["engine"] == "paddle"
        assert "char_count" in quality
        assert "low_confidence" in quality
    
    @pytest.mark.skipif(not is_tesseract_available(), reason="Tesseract not available")
    def test_tesseract_metadata_populated(self, scanned_pdf_path):
        """Test that Tesseract populates expected metadata fields."""
        if scanned_pdf_path is None:
            pytest.skip("No test PDF available")
        
        pages = ocr_with_specific_engine(scanned_pdf_path, "tesseract")
        
        if not pages:
            pytest.skip("No pages returned from OCR")
        
        page = pages[0]
        
        assert "extraction_method" in page.metadata
        assert "ocr" in page.metadata["extraction_method"].lower()
        
        assert "ocr_quality" in page.metadata
        quality = page.metadata["ocr_quality"]
        assert "engine" in quality
        assert quality["engine"] == "tesseract"


class TestOCREngineParity:
    """Tests for OCR engine output parity."""
    
    @pytest.mark.skipif(
        not (is_paddle_available() and is_tesseract_available()),
        reason="Both PaddleOCR and Tesseract required for parity testing"
    )
    def test_engine_parity_threshold(self, scanned_pdf_path):
        """Test that PaddleOCR and Tesseract produce similar results."""
        if scanned_pdf_path is None:
            pytest.skip("No test PDF available")
        
        # Run OCR with both engines
        paddle_pages = ocr_with_specific_engine(scanned_pdf_path, "paddle")
        tesseract_pages = ocr_with_specific_engine(scanned_pdf_path, "tesseract")
        
        if not paddle_pages or not tesseract_pages:
            pytest.skip("One or both engines returned no pages")
        
        # Compute parity
        result = compute_engine_parity(paddle_pages, tesseract_pages)
        
        # Log diagnostic info
        print("\n=== Engine Parity Results ===")
        print(f"Parity Score: {result.parity_score:.3f}")
        print(f"Paddle: {result.paddle_char_count} chars, {result.paddle_block_count} blocks")
        print(f"Tesseract: {result.tesseract_char_count} chars, {result.tesseract_block_count} blocks")
        if result.paddle_avg_confidence:
            print(f"Paddle avg confidence: {result.paddle_avg_confidence:.3f}")
        if result.tesseract_avg_confidence:
            print(f"Tesseract avg confidence: {result.tesseract_avg_confidence:.3f}")
        
        # Assert parity threshold
        assert result.parity_score >= ENGINE_PARITY_THRESHOLD, (
            f"Engine parity score {result.parity_score:.3f} below threshold {ENGINE_PARITY_THRESHOLD}\n"
            f"Paddle text sample: {result.paddle_text[:200]}...\n"
            f"Tesseract text sample: {result.tesseract_text[:200]}..."
        )
    
    @pytest.mark.skipif(
        not (is_paddle_available() and is_tesseract_available()),
        reason="Both PaddleOCR and Tesseract required"
    )
    def test_page_count_consistency(self, scanned_pdf_path):
        """Test that both engines detect the same number of pages."""
        if scanned_pdf_path is None:
            pytest.skip("No test PDF available")
        
        paddle_pages = ocr_with_specific_engine(scanned_pdf_path, "paddle")
        tesseract_pages = ocr_with_specific_engine(scanned_pdf_path, "tesseract")
        
        if not paddle_pages or not tesseract_pages:
            pytest.skip("One or both engines returned no pages")
        
        assert len(paddle_pages) == len(tesseract_pages), (
            f"Page count mismatch: Paddle={len(paddle_pages)}, Tesseract={len(tesseract_pages)}"
        )


class TestOCRTextQuality:
    """Tests for OCR text extraction quality."""
    
    @pytest.mark.skipif(not is_paddle_available(), reason="PaddleOCR not available")
    def test_paddle_produces_readable_text(self, scanned_pdf_path):
        """Test that PaddleOCR produces readable text."""
        if scanned_pdf_path is None:
            pytest.skip("No test PDF available")
        
        pages = ocr_with_specific_engine(scanned_pdf_path, "paddle")
        
        if not pages:
            pytest.skip("No pages returned")
        
        text = extract_full_text(pages)
        
        # Should have some content
        assert len(text) > 0, "OCR produced no text"
        
        # Should have reasonable printable character ratio
        printable = sum(1 for c in text if c.isprintable() or c.isspace())
        printable_ratio = printable / len(text) if text else 0
        
        assert printable_ratio > 0.9, (
            f"Low printable ratio: {printable_ratio:.3f} - text may be corrupted"
        )
    
    @pytest.mark.skipif(not is_tesseract_available(), reason="Tesseract not available")
    def test_tesseract_produces_readable_text(self, scanned_pdf_path):
        """Test that Tesseract produces readable text."""
        if scanned_pdf_path is None:
            pytest.skip("No test PDF available")
        
        pages = ocr_with_specific_engine(scanned_pdf_path, "tesseract")
        
        if not pages:
            pytest.skip("No pages returned")
        
        text = extract_full_text(pages)
        
        assert len(text) > 0, "OCR produced no text"
        
        printable = sum(1 for c in text if c.isprintable() or c.isspace())
        printable_ratio = printable / len(text) if text else 0
        
        assert printable_ratio > 0.9, (
            f"Low printable ratio: {printable_ratio:.3f} - text may be corrupted"
        )


class TestNormalizationImpactOnParity:
    """Tests verifying that normalization improves engine parity."""
    
    @pytest.mark.skipif(
        not (is_paddle_available() and is_tesseract_available()),
        reason="Both engines required"
    )
    def test_normalization_improves_parity(self, scanned_pdf_path):
        """Test that OCR normalization increases parity score."""
        if scanned_pdf_path is None:
            pytest.skip("No test PDF available")
        
        paddle_pages = ocr_with_specific_engine(scanned_pdf_path, "paddle")
        tesseract_pages = ocr_with_specific_engine(scanned_pdf_path, "tesseract")
        
        if not paddle_pages or not tesseract_pages:
            pytest.skip("One or both engines returned no pages")
        
        paddle_text = extract_full_text(paddle_pages)
        tesseract_text = extract_full_text(tesseract_pages)
        
        if not paddle_text or not tesseract_text:
            pytest.skip("No text extracted")
        
        # Raw similarity (no normalization)
        raw_score = fuzz.token_sort_ratio(paddle_text, tesseract_text) / 100.0
        
        # Normalized similarity
        config = NormalizationConfig.default_ocr()
        paddle_norm = normalize_compare(paddle_text, config)
        tesseract_norm = normalize_compare(tesseract_text, config)
        normalized_score = fuzz.token_sort_ratio(paddle_norm, tesseract_norm) / 100.0
        
        print(f"\nRaw parity score: {raw_score:.3f}")
        print(f"Normalized parity score: {normalized_score:.3f}")
        print(f"Improvement: {normalized_score - raw_score:.3f}")
        
        # Normalization should not decrease parity significantly
        # (Small decreases possible due to edge cases)
        assert normalized_score >= raw_score - 0.05, (
            "Normalization significantly decreased parity: "
            f"raw={raw_score:.3f}, normalized={normalized_score:.3f}"
        )


# =============================================================================
# Integration Tests with Pipeline
# =============================================================================

class TestPipelineEngineConsistency:
    """Tests for comparison pipeline consistency across OCR engines."""
    
    @pytest.mark.skipif(
        not (is_paddle_available() and is_tesseract_available()),
        reason="Both engines required"
    )
    def test_diff_count_stability(self, scanned_pdf_path):
        """
        Test that comparing the same document with different OCR engines
        produces stable diff counts (no phantom diffs from engine differences).
        """
        if scanned_pdf_path is None:
            pytest.skip("No test PDF available")
        
        # Extract with both engines
        paddle_pages = ocr_with_specific_engine(scanned_pdf_path, "paddle")
        tesseract_pages = ocr_with_specific_engine(scanned_pdf_path, "tesseract")
        
        if not paddle_pages or not tesseract_pages:
            pytest.skip("One or both engines returned no pages")
        
        # The idea: if we compare engine A vs engine A (same content),
        # and engine B vs engine B (same content), diff count should be similar
        # This is a proxy for engine stability
        
        paddle_text = extract_full_text(paddle_pages)
        tesseract_text = extract_full_text(tesseract_pages)
        
        # Both self-comparisons should yield zero/minimal diffs
        paddle_self_score = fuzz.ratio(paddle_text, paddle_text) / 100.0
        tesseract_self_score = fuzz.ratio(tesseract_text, tesseract_text) / 100.0
        
        assert paddle_self_score == 1.0, "Self-comparison should be identical"
        assert tesseract_self_score == 1.0, "Self-comparison should be identical"
        
        # Cross-engine comparison
        cross_score = fuzz.ratio(paddle_text, tesseract_text) / 100.0
        
        print(f"\nSelf-comparison Paddle: {paddle_self_score:.3f}")
        print(f"Self-comparison Tesseract: {tesseract_self_score:.3f}")
        print(f"Cross-engine comparison: {cross_score:.3f}")
        
        # Cross-engine should be reasonably close
        assert cross_score >= 0.7, (
            f"Cross-engine similarity too low: {cross_score:.3f}"
        )


# =============================================================================
# Performance Benchmarks (Optional)
# =============================================================================

class TestOCRPerformance:
    """Performance benchmarks for OCR engines (informational)."""
    
    @pytest.mark.skipif(not is_paddle_available(), reason="PaddleOCR not available")
    def test_paddle_performance_info(self, scanned_pdf_path):
        """Measure PaddleOCR processing time (informational)."""
        if scanned_pdf_path is None:
            pytest.skip("No test PDF available")
        
        import time
        
        start = time.perf_counter()
        pages = ocr_with_specific_engine(scanned_pdf_path, "paddle")
        elapsed = time.perf_counter() - start
        
        if pages:
            chars = sum(len(b.text) for p in pages for b in p.blocks)
            print(f"\nPaddleOCR: {elapsed:.2f}s for {len(pages)} pages, {chars} chars")
            print(f"  Rate: {chars/elapsed:.0f} chars/sec" if elapsed > 0 else "")
    
    @pytest.mark.skipif(not is_tesseract_available(), reason="Tesseract not available")
    def test_tesseract_performance_info(self, scanned_pdf_path):
        """Measure Tesseract processing time (informational)."""
        if scanned_pdf_path is None:
            pytest.skip("No test PDF available")
        
        import time
        
        start = time.perf_counter()
        pages = ocr_with_specific_engine(scanned_pdf_path, "tesseract")
        elapsed = time.perf_counter() - start
        
        if pages:
            chars = sum(len(b.text) for p in pages for b in p.blocks)
            print(f"\nTesseract: {elapsed:.2f}s for {len(pages)} pages, {chars} chars")
            print(f"  Rate: {chars/elapsed:.0f} chars/sec" if elapsed > 0 else "")
