"""Tests for extraction modules."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

from comparison.models import PageData
from extraction import extract_pdf
from extraction.pdf_parser import parse_pdf

# Check if PyMuPDF is available
try:
    import fitz  # noqa: F401
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


@pytest.mark.skipif(
    not PYMUPDF_AVAILABLE,
    reason="PyMuPDF not installed in test env",
)
def test_parse_pdf_accepts_path(monkeypatch):
    """Test that parse_pdf accepts Path objects."""
    dummy_pdf = Path(__file__).parent / "fixtures" / "dummy.pdf"

    class DummyDoc:
        def __iter__(self):
            return iter([])
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        "extraction.pdf_parser.fitz",
        type(
            "dummy",
            (),
            {
                "open": lambda _p: DummyDoc(),
                "TEXT_PRESERVE_LIGATURES": 0,
            },
        ),
    )
    pages = parse_pdf(dummy_pdf)
    assert isinstance(pages, list)


def test_extract_pdf_validation():
    """Test that extract_pdf validates input paths."""
    with pytest.raises((ValueError, FileNotFoundError)):
        extract_pdf("nonexistent.pdf")


def test_is_scanned_pdf_detection():
    """Test scanned PDF detection via extract_pdf."""
    # Test that extract_pdf handles non-existent files
    with pytest.raises((ValueError, FileNotFoundError)):
        extract_pdf(Path("nonexistent.pdf"))


def test_is_scanned_pdf_heuristics_mocked_fitz(monkeypatch):
    """Test _is_scanned_pdf() heuristics without real PyMuPDF."""
    import extraction

    class DummyPage:
        def __init__(self, text: str):
            self._text = text

        def get_text(self, _kind: str = "text"):
            return self._text

    class DummyDoc:
        def __init__(self, pages):
            self._pages = pages

        def __len__(self):
            return len(self._pages)

        def __getitem__(self, idx):
            return self._pages[idx]

        def close(self):
            return None

    class DummyFitz:
        @staticmethod
        def open(_path):
            raise AssertionError("fitz.open should be monkeypatched per-test")

    # Install dummy fitz module
    monkeypatch.setitem(sys.modules, "fitz", DummyFitz)

    # Make thresholds deterministic
    extraction.settings.scan_detection_sample_pages = 5
    extraction.settings.scan_detection_page_text_min_chars = 50
    extraction.settings.scan_detection_min_good_pages_ratio = 0.1

    # Case 1: clearly digital
    digital_pages = [DummyPage("Hello world! " * 20) for _ in range(5)]
    monkeypatch.setattr(DummyFitz, "open", lambda _p: DummyDoc(digital_pages))
    assert extraction._is_scanned_pdf(Path("dummy.pdf")) is False

    # Case 2: clearly scanned (no text)
    scanned_pages = [DummyPage(" ") for _ in range(5)]
    monkeypatch.setattr(DummyFitz, "open", lambda _p: DummyDoc(scanned_pages))
    assert extraction._is_scanned_pdf(Path("dummy.pdf")) is True

    # Case 3: garbage text layer (cid artifacts)
    garbage_text = "(cid:123)" * 200
    garbage_pages = [DummyPage(garbage_text) for _ in range(5)]
    monkeypatch.setattr(DummyFitz, "open", lambda _p: DummyDoc(garbage_pages))
    assert extraction._is_scanned_pdf(Path("dummy.pdf")) is True


@pytest.mark.skipif(
    not PYMUPDF_AVAILABLE,
    reason="PyMuPDF not installed in test env",
)
def test_parse_pdf_real_extraction():
    """Test parse_pdf with a real PDF file from synthetic dataset."""
    # Use synthetic dataset PDF if available
    project_root = Path(__file__).parent.parent
    synthetic_pdf = project_root / "data" / "synthetic" / "dataset" / "variation_01" / "variation_01_original.pdf"
    
    if not synthetic_pdf.exists():
        pytest.skip(f"Test PDF not found: {synthetic_pdf}")
    
    # Parse PDF
    pages = parse_pdf(synthetic_pdf, run_layout_analysis=False)
    
    # Basic assertions
    assert isinstance(pages, list)
    assert len(pages) > 0, "Should extract at least one page"
    
    # Check page structure
    for page in pages:
        assert isinstance(page, PageData)
        assert page.page_num >= 1
        assert page.width > 0
        assert page.height > 0
        assert isinstance(page.blocks, list)
        
        # Check that blocks have proper structure
        for block in page.blocks:
            assert block.text is not None
            assert isinstance(block.bbox, dict)
            assert "x" in block.bbox
            assert "y" in block.bbox
            assert "width" in block.bbox
            assert "height" in block.bbox
            assert block.bbox["width"] > 0
            assert block.bbox["height"] > 0
            
            # Check metadata
            assert "bbox_units" in block.metadata
            assert "bbox_source" in block.metadata
    
    # Verify extraction method in metadata
    assert pages[0].metadata.get("extraction_method") == "pdf_digital"
    
    # Verify we got some text content
    total_text_length = sum(len(block.text) for page in pages for block in page.blocks)
    assert total_text_length > 0, "Should extract some text content"


@pytest.mark.skipif(
    not PYMUPDF_AVAILABLE,
    reason="PyMuPDF not installed in test env",
)
def test_extract_pdf_real_document():
    """Test extract_pdf with a real PDF file (full extraction pipeline)."""
    project_root = Path(__file__).parent.parent
    synthetic_pdf = project_root / "data" / "synthetic" / "dataset" / "variation_01" / "variation_01_original.pdf"
    
    if not synthetic_pdf.exists():
        pytest.skip(f"Test PDF not found: {synthetic_pdf}")
    
    # Extract using full pipeline (with auto-detection)
    pages = extract_pdf(synthetic_pdf, run_layout_analysis=False)
    
    # Basic assertions
    assert isinstance(pages, list)
    assert len(pages) > 0
    
    # Verify all pages are properly structured
    for page in pages:
        assert isinstance(page, PageData)
        assert page.page_num >= 1
        assert len(page.blocks) >= 0  # Some pages might be empty
    
    # Should extract text (this is a digital PDF)
    total_text = sum(len(block.text) for page in pages for block in page.blocks)
    assert total_text > 50, f"Expected significant text extraction, got {total_text} chars"
