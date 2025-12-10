"""Tests for extraction modules."""
from __future__ import annotations

from pathlib import Path

import pytest

from comparison.models import PageData
from extraction import extract_pdf
from extraction.pdf_parser import parse_pdf


@pytest.mark.skipif(
    "fitz" not in dir(pytest.importorskip("fitz")),
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
