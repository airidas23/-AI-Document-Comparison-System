"""Test script for OCR bounding box visualization.

Note: This is a runnable script (not a pytest test). It must add the repository
root (parent of `tests/`) to `sys.path` so imports like `extraction.*` work when
executed directly.
"""

import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
tests_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from extraction.ocr_visualizer import visualize_ocr_on_pdf_page


def main():
    # Use a sample PDF from the dataset
    pdf_path = repo_root / "data/synthetic/dataset/base_document.pdf"
    
    if not pdf_path.exists():
        print(f"[ERROR] Sample PDF not found at: {pdf_path}")
        sys.exit(1)
    
    output_path = tests_dir / "ocr_viz_output.png"
    
    print(f"[INFO] Running OCR visualization on: {pdf_path}")
    print(f"[INFO] Using OCR engine: tesseract")
    
    try:
        result_img = visualize_ocr_on_pdf_page(
            pdf_path=str(pdf_path),
            page_num=0,
            ocr_engine="tesseract",
            output_path=str(output_path),
            show_text=True,
            show_confidence=True,
        )
        print(f"[SUCCESS] Output saved to: {output_path}")
        print(f"[INFO] Image size: {result_img.size}")
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
