#!/usr/bin/env python3
"""Detailed timing breakdown for OCR engines."""
import time
import fitz
from pathlib import Path
from PIL import Image
import pytesseract
from pytesseract import Output

def test_tesseract_timing():
    test_pdf = Path('data/synthetic/test_output_20p/variation_01/variation_01_original_scanned.pdf')
    doc = fitz.open(test_pdf)

    print('=== TESSERACT DETAILED TIMING BREAKDOWN ===')
    print(f'PDF: {test_pdf}')
    print(f'Pages: {len(doc)}')
    print(f'DPI: 100')
    print()

    # Run 3 times and collect stats
    for run in range(1, 4):
        print(f'--- Run {run} ---')
        total_render = 0
        total_ocr = 0
        total_parse = 0
        total_blocks = 0
        
        run_start = time.time()
        
        for page in doc:
            # 1. Render
            t0 = time.time()
            pix = page.get_pixmap(dpi=100)
            img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            render_time = time.time() - t0
            total_render += render_time
            
            # 2. OCR
            t0 = time.time()
            ocr_data = pytesseract.image_to_data(img, lang='eng', output_type=Output.DICT, config='--psm 6 --oem 3')
            ocr_time = time.time() - t0
            total_ocr += ocr_time
            
            # 3. Parse results
            t0 = time.time()
            blocks = [t for t in ocr_data['text'] if t.strip()]
            parse_time = time.time() - t0
            total_parse += parse_time
            total_blocks += len(blocks)
        
        run_total = time.time() - run_start
        
        print(f'  Render:  {total_render:.2f}s ({total_render/len(doc):.3f}s/page)')
        print(f'  OCR:     {total_ocr:.2f}s ({total_ocr/len(doc):.3f}s/page)')
        print(f'  Parse:   {total_parse:.4f}s')
        print(f'  TOTAL:   {run_total:.2f}s ({run_total/len(doc):.3f}s/page)')
        print(f'  Blocks:  {total_blocks}')
        print()

    doc.close()


def test_paddle_timing():
    from extraction.paddle_ocr_engine import _get_ocr
    import numpy as np
    
    test_pdf = Path('data/synthetic/test_output_20p/variation_01/variation_01_original_scanned.pdf')
    doc = fitz.open(test_pdf)

    print('=== PADDLEOCR DETAILED TIMING BREAKDOWN ===')
    print(f'PDF: {test_pdf}')
    print(f'Pages: {len(doc)}')
    print(f'DPI: 100')
    print()
    
    # Warm up - load model
    print('Loading PaddleOCR model...')
    t0 = time.time()
    ocr = _get_ocr()
    print(f'Model loaded in {time.time() - t0:.2f}s')
    print()

    # Run 3 times and collect stats
    for run in range(1, 4):
        print(f'--- Run {run} ---')
        total_render = 0
        total_ocr = 0
        total_parse = 0
        total_blocks = 0
        
        run_start = time.time()
        
        for page in doc:
            # 1. Render
            t0 = time.time()
            pix = page.get_pixmap(dpi=100)
            img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            img_array = np.array(img)
            render_time = time.time() - t0
            total_render += render_time
            
            # 2. OCR
            t0 = time.time()
            ocr_result = ocr.predict(img_array)
            ocr_time = time.time() - t0
            total_ocr += ocr_time
            
            # 3. Parse results (count blocks)
            t0 = time.time()
            blocks = 0
            if ocr_result:
                for result in ocr_result:
                    if isinstance(result, dict):
                        blocks += len(result.get('rec_texts', []))
            parse_time = time.time() - t0
            total_parse += parse_time
            total_blocks += blocks
        
        run_total = time.time() - run_start
        
        print(f'  Render:  {total_render:.2f}s ({total_render/len(doc):.3f}s/page)')
        print(f'  OCR:     {total_ocr:.2f}s ({total_ocr/len(doc):.3f}s/page)')
        print(f'  Parse:   {total_parse:.4f}s')
        print(f'  TOTAL:   {run_total:.2f}s ({run_total/len(doc):.3f}s/page)')
        print(f'  Blocks:  {total_blocks}')
        print()

    doc.close()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'paddle':
        test_paddle_timing()
    else:
        test_tesseract_timing()
