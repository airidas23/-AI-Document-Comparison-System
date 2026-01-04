import pytest
from unittest.mock import MagicMock, patch
from extraction.tesseract_ocr_engine import ocr_pdf
from comparison.models import TextBlock

def check_bbox_overlap(b1, b2):
    """Check if two bboxes overlap efficiently."""
    x1_min, y1_min = b1.bbox['x'], b1.bbox['y']
    x1_max, y1_max = x1_min + b1.bbox['width'], y1_min + b1.bbox['height']
    
    x2_min, y2_min = b2.bbox['x'], b2.bbox['y']
    x2_max, y2_max = x2_min + b2.bbox['width'], y2_min + b2.bbox['height']
    
    # Check for separation
    if x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min:
        return 0.0
        
    # Calculate overlap area
    ix_min = max(x1_min, x2_min)
    iy_min = max(y1_min, y2_min)
    ix_max = min(x1_max, x2_max)
    iy_max = min(y1_max, y2_max)
    
    return max(0, ix_max - ix_min) * max(0, iy_max - iy_min)

def test_tesseract_bboxes_do_not_overlap():
    """Verify that Tesseract OCR output does not contain overlapping blocks."""
    
    # Use the specific PDF mentioned by the user
    pdf_path = "data/test/lt_scan_demo/DOC_A_scanned.pdf"
    
    # Mock settings to ensure known state
    mock_settings = MagicMock()
    mock_settings.tesseract_granularity = "block"
    mock_settings.tesseract_render_dpi = 300
    mock_settings.num_workers = 1
    mock_settings.tesseract_lang = "lit+eng"
    mock_settings.tesseract_config_string = ""
    mock_settings.tesseract_disable_dawg = False
    mock_settings.tesseract_psm_fallback_enabled = False
    mock_settings.tesseract_psm_mode = 3
    
    with patch('config.settings.settings', mock_settings):
        pages = ocr_pdf(pdf_path)
        
        assert len(pages) > 0, "Should extract pages"
        
        for page in pages:
            blocks = page.blocks
            # Check every pair for overlaps
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    area = check_bbox_overlap(blocks[i], blocks[j])
                    
                    # Allow very small overlaps (e.g. 1-2 pixels due to rounding or tight layout)
                    # But large overlaps indicate a problem
                    msg = f"Blocks overlap on page {page.page_num}: '{blocks[i].text[:20]}' vs '{blocks[j].text[:20]}'"
                    assert area < 20.0, msg  # 20 sq px is small threshold
                    
def test_tesseract_word_granularity_overlaps():
    """Verify no overlaps with word granularity."""
    pdf_path = "data/test/lt_scan_demo/DOC_A_scanned.pdf"
    
    mock_settings = MagicMock()
    mock_settings.tesseract_granularity = "word"
    mock_settings.tesseract_render_dpi = 300
    mock_settings.num_workers = 1
    mock_settings.tesseract_lang = "lit+eng"
    mock_settings.tesseract_config_string = ""
    
    with patch('config.settings.settings', mock_settings):
        pages = ocr_pdf(pdf_path)
        for page in pages:
            blocks = page.blocks
            # Check consecutive words explicitly (optimization)
            # Tesseract sorts roughly reading order, so overlaps likely close in index
            for i in range(len(blocks)):
                # Check against all others to be sure
                for j in range(i + 1, len(blocks)):
                     # Optimization: if y-distance is huge, skip
                    if abs(blocks[i].bbox['y'] - blocks[j].bbox['y']) > 50:
                        continue
                        
                    area = check_bbox_overlap(blocks[i], blocks[j])
                    assert area < 20.0, f"Words overlap: '{blocks[i].text}' vs '{blocks[j].text}'"
