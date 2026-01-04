import sys
import os
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from extraction.tesseract_ocr_engine import ocr_pdf
from comparison.models import TextBlock

def check_overlaps(blocks):
    overlaps = []
    for i, b1 in enumerate(blocks):
        for j, b2 in enumerate(blocks):
            if i >= j: continue
            
            # Simple bbox intersection check
            x1_min, y1_min = b1.bbox['x'], b1.bbox['y']
            x1_max, y1_max = x1_min + b1.bbox['width'], y1_min + b1.bbox['height']
            
            x2_min, y2_min = b2.bbox['x'], b2.bbox['y']
            x2_max, y2_max = x2_min + b2.bbox['width'], y2_min + b2.bbox['height']
            
            intersect = not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)
            if intersect:
                # Calculate Intersection Area
                ix_min = max(x1_min, x2_min)
                iy_min = max(y1_min, y2_min)
                ix_max = min(x1_max, x2_max)
                iy_max = min(y1_max, y2_max)
                area = max(0, ix_max - ix_min) * max(0, iy_max - iy_min)
                
                if area > 10: # Ignore tiny pixel overlaps
                     overlaps.append((i, j, area, b1.text[:30], b2.text[:30]))
    return overlaps

def run_test(granularity="block", psm_mode=3):
    print(f"\n--- Testing Granularity: {granularity}, PSM: {psm_mode} ---")
    
    # Mock settings
    mock_settings = MagicMock()
    mock_settings.tesseract_granularity = granularity
    mock_settings.tesseract_psm_mode = psm_mode
    mock_settings.tesseract_render_dpi = 300
    mock_settings.num_workers = 1
    mock_settings.tesseract_lang = "lit+eng"
    mock_settings.tesseract_config_string = ""
    mock_settings.tesseract_disable_dawg = False
    mock_settings.tesseract_psm_fallback_enabled = False
    
    with patch('config.settings.settings', mock_settings):
        try:
            pages = ocr_pdf("data/test/lt_scan_demo/DOC_A_scanned.pdf")
            print(f"Extracted {len(pages)} pages.")
            for p in pages:
                print(f"Page {p.page_num}: {len(p.blocks)} blocks")
                
                # Print sample blocks
                print("  Sample blocks:")
                for b in p.blocks[:3]:
                    print(f"    [{b.bbox['x']:.0f}, {b.bbox['y']:.0f}, w={b.bbox['width']:.0f}, h={b.bbox['height']:.0f}] '{b.text[:20]}...'")

                overlaps = check_overlaps(p.blocks)
                if overlaps:
                    print(f"FAILED: Found {len(overlaps)} overlaps on Page {p.page_num}:")
                    for idx1, idx2, area, txt1, txt2 in overlaps[:5]:
                        print(f"  Overlap: '{txt1.strip()}' vs '{txt2.strip()}' (Area: {area:.2f})")
                else:
                    print("PASSED: No significant overlaps found.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_test("block", 3) # Auto
    run_test("block", 6) # Single block
    run_test("block", 4) # Single column 
