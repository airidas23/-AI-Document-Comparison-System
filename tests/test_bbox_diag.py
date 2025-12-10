"""Diagnostic script to check OCR bounding box coordinates."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_bbox_scaling():
    """Test bounding box coordinates from OCR engines."""
    pdf_path = "data/synthetic/dataset/variation_01/variation_01_original.pdf"
    
    if not Path(pdf_path).exists():
        print(f"PDF not found: {pdf_path}")
        return
    
    # Test Paddle OCR
    print("=" * 60)
    print("PADDLE OCR TEST")
    print("=" * 60)
    try:
        from extraction.paddle_ocr_engine import ocr_pdf as paddle_ocr
        pages = paddle_ocr(pdf_path)
        if pages:
            page = pages[0]
            print(f"Page dimensions (PDF points): {page.width} x {page.height}")
            print(f"Number of blocks: {len(page.blocks)}")
            if page.blocks:
                block = page.blocks[0]
                print(f"First block bbox: {block.bbox}")
                print(f"First block text: {block.text[:50]}...")
    except Exception as e:
        print(f"Paddle OCR failed: {e}")
    
    # Test visualization scaling
    print("\n" + "=" * 60)
    print("VISUALIZATION SCALING TEST")
    print("=" * 60)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page_obj = doc[0]
        
        # PDF dimensions
        pdf_width = page_obj.rect.width
        pdf_height = page_obj.rect.height
        print(f"PDF page rect: {pdf_width} x {pdf_height} points")
        
        # Render at 300 DPI with 2x matrix (what visualizer does)
        pix = page_obj.get_pixmap(dpi=300, matrix=fitz.Matrix(2, 2))
        print(f"Visualization render: {pix.width} x {pix.height} pixels")
        
        # Render at 150 DPI (what OCR engines do)
        pix_ocr = page_obj.get_pixmap(dpi=150)
        print(f"OCR render (150 DPI): {pix_ocr.width} x {pix_ocr.height} pixels")
        
        # Calculate scale factors
        scale_viz_to_pdf = pdf_width / pix.width
        scale_ocr_to_pdf = 72.0 / 150.0
        scale_pdf_to_viz = pix.width / pdf_width
        
        print(f"\nScale factors:")
        print(f"  Viz pixels -> PDF points: {scale_viz_to_pdf:.4f}")
        print(f"  OCR 150 DPI -> PDF points (72 DPI): {scale_ocr_to_pdf:.4f}")
        print(f"  PDF points -> Viz pixels: {scale_pdf_to_viz:.4f}")
        
        doc.close()
    except Exception as e:
        print(f"Scaling test failed: {e}")

if __name__ == "__main__":
    test_bbox_scaling()
