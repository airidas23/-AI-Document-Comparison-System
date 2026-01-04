#!/usr/bin/env python3
"""Diagnose extraction timing - detect multiple OCR calls."""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings


def diagnose_extraction(pdf_path: str, engine: str = "tesseract") -> dict:
    """
    Diagnose extraction with detailed sub-timing.
    
    Returns breakdown of:
    - extract_pdf time (main OCR)
    - extract_lines time (should now be instant - reuses blocks)
    - layout analysis time
    """
    import fitz
    from extraction import extract_pdf
    from extraction.line_extractor import extract_lines
    from extraction.layout_analyzer import analyze_layout
    
    path = Path(pdf_path)
    doc = fitz.open(path)
    total_pages = len(doc)
    doc.close()
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION DIAGNOSIS")
    print(f"{'='*60}")
    print(f"PDF: {path.name}")
    print(f"Pages: {total_pages}")
    print(f"OCR Engine: {engine}")
    print(f"{'='*60}\n")
    
    # Configure engine
    settings.ocr_engine = engine
    if engine == "tesseract":
        settings.tesseract_render_dpi = 100
    else:
        settings.paddle_render_dpi = 100
    
    timings = {}
    
    # 1. Main extraction (extract_pdf)
    print("[1] Running extract_pdf()...")
    t1 = time.time()
    pages = extract_pdf(path, force_ocr=True)
    t2 = time.time()
    timings["extract_pdf"] = t2 - t1
    total_blocks = sum(len(p.blocks) for p in pages)
    print(f"    → {len(pages)} pages, {total_blocks} blocks")
    print(f"    → Time: {timings['extract_pdf']:.2f}s ({timings['extract_pdf']/total_pages:.2f}s/page)")
    
    # Check which engine was actually used
    if pages:
        method = pages[0].metadata.get("extraction_method", "unknown")
        ocr_engine = pages[0].metadata.get("ocr_engine_used", "unknown")
        print(f"    → Method: {method}, Engine: {ocr_engine}")
    
    # 2. Line extraction - NOW should be instant (reuses blocks)
    print("\n[2] Running extract_lines(pages)...")
    t3 = time.time()
    pages = extract_lines(pages)  # Pass pages, not path!
    t4 = time.time()
    timings["extract_lines"] = t4 - t3
    total_lines = sum(len(p.lines) for p in pages)
    print(f"    → {len(pages)} pages, {total_lines} lines")
    print(f"    → Time: {timings['extract_lines']:.2f}s")
    
    # Check line extraction method
    if pages:
        line_method = pages[0].metadata.get("line_extraction_method", "unknown")
        print(f"    → Method: {line_method}")
        
        if line_method == "from_blocks":
            print(f"    ✓ Line extraction reused blocks (no additional OCR)")
        elif "ocr" in line_method.lower():
            print(f"    ⚠️  WARNING: Line extraction used OCR again!")
    
    # 3. Layout analysis
    print("\n[3] Running analyze_layout()...")
    t5 = time.time()
    layout_pages = analyze_layout(path)
    t6 = time.time()
    timings["layout_analysis"] = t6 - t5
    print(f"    → {len(layout_pages)} pages")
    print(f"    → Time: {timings['layout_analysis']:.2f}s")
    
    # Summary
    total_time = sum(timings.values())
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY")
    print(f"{'='*60}")
    for name, t in timings.items():
        pct = (t / total_time * 100) if total_time > 0 else 0
        print(f"{name:20s}: {t:6.2f}s ({pct:5.1f}%)")
    print(f"{'='*60}")
    print(f"{'TOTAL':20s}: {total_time:6.2f}s ({total_time/total_pages:.2f}s/page)")
    print(f"{'='*60}\n")
    
    # Diagnosis
    print("DIAGNOSIS:")
    if timings["extract_lines"] > 1.0:
        print("  ❌ extract_lines() is still slow!")
    else:
        print("  ✓ extract_lines() is fast - reusing blocks correctly")
    
    return timings


def main():
    """Run diagnosis on test PDF."""
    # Default test path
    test_pdf = project_root / "data" / "synthetic" / "test_output_20p" / "variation_01" / "variation_01_original_scanned.pdf"
    
    if len(sys.argv) > 1:
        test_pdf = Path(sys.argv[1])
    
    if not test_pdf.exists():
        print(f"ERROR: PDF not found: {test_pdf}")
        return 1
    
    # Test with Tesseract
    print("\n" + "="*70)
    print("TESTING WITH TESSERACT")
    print("="*70)
    diagnose_extraction(str(test_pdf), engine="tesseract")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
