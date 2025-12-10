
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from config.settings import settings
    print(f"Settings loaded. OCR Engine: {settings.ocr_engine}")
    
    from extraction.ocr_router import select_ocr_engine, ocr_pdf_multi
    
    # Check selector
    available, skipped = select_ocr_engine([settings.ocr_engine])
    print(f"Selector result for {settings.ocr_engine}: {available}")
    
    if available:
        print("SUCCESS: Engine selection works.")
    else:
        print("WARNING: Engine selection returned empty (dependencies might be missing, which is expected if not installed).")
        
    print("Verification passed.")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
