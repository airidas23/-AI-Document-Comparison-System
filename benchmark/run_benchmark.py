#!/usr/bin/env python3
"""
OCR Engine Benchmark - PyMuPDF vs Tesseract vs PaddleOCR
"""
import time
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_benchmark():
    digital_pdf = Path("data/synthetic/dataset/variation_01/variation_01_original.pdf")
    scanned_pdf = Path("data/synthetic/test_scanned_dataset/variation_01/variation_01_original_scanned.pdf")

    results = {"digital_pdf": {}, "scanned_pdf": {}}

    print("=" * 70)
    print("            OCR ENGINE BENCHMARK")
    print("=" * 70)

    # 1. PyMuPDF
    print("\n[1/5] PyMuPDF - Digital PDF...")
    import fitz
    t0 = time.time()
    doc = fitz.open(str(digital_pdf))
    text = ""
    blocks = 0
    for page in doc:
        text += page.get_text()
        blocks += len(page.get_text("dict")["blocks"])
    doc.close()
    dt = time.time() - t0
    results["digital_pdf"]["pymupdf"] = {"time": round(dt, 3), "chars": len(text), "blocks": blocks}
    print(f"   Time: {dt:.3f}s | Chars: {len(text):,} | Blocks: {blocks}")

    # 2. Tesseract - Digital
    print("\n[2/5] Tesseract - Digital PDF...")
    from extraction.tesseract_ocr_engine import ocr_pdf as tesseract_ocr_pdf
    t0 = time.time()
    pages = tesseract_ocr_pdf(digital_pdf)
    dt = time.time() - t0
    chars = sum(len(b.text or "") for p in pages for b in (p.blocks or []))
    blks = sum(len(p.blocks or []) for p in pages)
    results["digital_pdf"]["tesseract"] = {"time": round(dt, 3), "chars": chars, "blocks": blks}
    print(f"   Time: {dt:.3f}s | Chars: {chars:,} | Blocks: {blks}")

    # 3. Tesseract - Scanned
    print("\n[3/5] Tesseract - Scanned PDF...")
    t0 = time.time()
    pages = tesseract_ocr_pdf(scanned_pdf)
    dt = time.time() - t0
    chars = sum(len(b.text or "") for p in pages for b in (p.blocks or []))
    blks = sum(len(p.blocks or []) for p in pages)
    results["scanned_pdf"]["tesseract"] = {"time": round(dt, 3), "chars": chars, "blocks": blks}
    print(f"   Time: {dt:.3f}s | Chars: {chars:,} | Blocks: {blks}")

    # 4. PaddleOCR - Digital
    print("\n[4/5] PaddleOCR - Digital PDF...")
    from extraction.paddle_ocr_engine import ocr_pdf as paddle_ocr_pdf
    t0 = time.time()
    pages = paddle_ocr_pdf(digital_pdf)
    dt = time.time() - t0
    chars = sum(len(b.text or "") for p in pages for b in (p.blocks or []))
    blks = sum(len(p.blocks or []) for p in pages)
    results["digital_pdf"]["paddle"] = {"time": round(dt, 3), "chars": chars, "blocks": blks}
    print(f"   Time: {dt:.3f}s | Chars: {chars:,} | Blocks: {blks}")

    # 5. PaddleOCR - Scanned
    print("\n[5/5] PaddleOCR - Scanned PDF...")
    t0 = time.time()
    pages = paddle_ocr_pdf(scanned_pdf)
    dt = time.time() - t0
    chars = sum(len(b.text or "") for p in pages for b in (p.blocks or []))
    blks = sum(len(p.blocks or []) for p in pages)
    results["scanned_pdf"]["paddle"] = {"time": round(dt, 3), "chars": chars, "blocks": blks}
    print(f"   Time: {dt:.3f}s | Chars: {chars:,} | Blocks: {blks}")

    # Summary
    print("\n" + "=" * 70)
    print("                    BENCHMARK RESULTS")
    print("=" * 70)

    print("\nDIGITAL PDF PERFORMANCE (1 page)")
    print("-" * 50)
    print(f"{'Engine':<12} {'Time (s)':<12} {'Chars':<12} {'Blocks':<12}")
    print("-" * 50)
    for eng in ["pymupdf", "tesseract", "paddle"]:
        d = results["digital_pdf"][eng]
        print(f"{eng:<12} {d['time']:<12} {d['chars']:<12} {d['blocks']:<12}")

    print("\nSCANNED PDF PERFORMANCE (1 page)")
    print("-" * 50)
    print(f"{'Engine':<12} {'Time (s)':<12} {'Chars':<12} {'Blocks':<12}")
    print("-" * 50)
    for eng in ["tesseract", "paddle"]:
        d = results["scanned_pdf"][eng]
        print(f"{eng:<12} {d['time']:<12} {d['chars']:<12} {d['blocks']:<12}")

    # Speed comparison
    pymupdf_t = results["digital_pdf"]["pymupdf"]["time"]
    tess_dig_t = results["digital_pdf"]["tesseract"]["time"]
    paddle_dig_t = results["digital_pdf"]["paddle"]["time"]
    tess_scan_t = results["scanned_pdf"]["tesseract"]["time"]
    paddle_scan_t = results["scanned_pdf"]["paddle"]["time"]

    print("\nSPEED COMPARISON")
    print("-" * 50)
    if pymupdf_t > 0 and tess_dig_t > 0:
        print(f"PyMuPDF vs Tesseract (digital): PyMuPDF {tess_dig_t/pymupdf_t:.0f}x faster")
    if pymupdf_t > 0 and paddle_dig_t > 0:
        print(f"PyMuPDF vs PaddleOCR (digital): PyMuPDF {paddle_dig_t/pymupdf_t:.0f}x faster")
    if tess_scan_t > 0 and paddle_scan_t > 0:
        ratio = paddle_scan_t/tess_scan_t
        if ratio > 1:
            print(f"Tesseract vs PaddleOCR (scanned): Tesseract {ratio:.1f}x faster")
        else:
            print(f"Tesseract vs PaddleOCR (scanned): PaddleOCR {1/ratio:.1f}x faster")

    # Character extraction accuracy comparison
    print("\nCHARACTER EXTRACTION COMPARISON (Digital PDF)")
    print("-" * 50)
    pymupdf_chars = results["digital_pdf"]["pymupdf"]["chars"]
    print(f"PyMuPDF (baseline):   {pymupdf_chars:,} chars")
    for eng in ["tesseract", "paddle"]:
        eng_chars = results["digital_pdf"][eng]["chars"]
        diff = eng_chars - pymupdf_chars
        pct = (eng_chars / pymupdf_chars * 100) if pymupdf_chars > 0 else 0
        print(f"{eng:<12}          {eng_chars:,} chars ({pct:.1f}% of baseline, diff: {diff:+,})")

    # Save results
    output_file = Path("benchmark/benchmark_results.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    run_benchmark()
