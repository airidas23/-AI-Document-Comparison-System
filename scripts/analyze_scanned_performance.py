#!/usr/bin/env python3
"""
Analyze OCR performance on scanned PDFs from test_output_20p.

Compares different OCR engines and shows timing + accuracy.
"""
import json
import time
from pathlib import Path
from collections import defaultdict

from pipeline.compare_pdfs import compare_pdfs


def analyze_variation(variation_dir: Path, ocr_engine: str):
    """Analyze a single variation with specified OCR engine."""
    variation_name = variation_dir.name
    original_pdf = variation_dir / f"{variation_name}_original_scanned.pdf"
    modified_pdf = variation_dir / f"{variation_name}_modified_scanned.pdf"
    
    if not original_pdf.exists() or not modified_pdf.exists():
        return None
    
    # Load ground truth
    change_log_path = variation_dir / f"{variation_name}_change_log.json"
    with open(change_log_path) as f:
        change_log = json.load(f)
    
    gt_pages = {c['page'] for c in change_log['changes']}
    
    # Run comparison
    start = time.time()
    result = compare_pdfs(str(original_pdf), str(modified_pdf), ocr_engine=ocr_engine)
    elapsed = time.time() - start
    
    # Group by page
    by_page = defaultdict(list)
    for diff in result.diffs:
        by_page[diff.page_num].append(diff)
    
    detected_pages = set(by_page.keys())
    
    tp = len(gt_pages & detected_pages)
    fp = len(detected_pages - gt_pages)
    fn = len(gt_pages - detected_pages)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    return {
        'variation': variation_name,
        'ocr_engine': ocr_engine,
        'elapsed': elapsed,
        'gt_pages': sorted(gt_pages),
        'detected_pages': sorted(detected_pages),
        'total_diffs': len(result.diffs),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    dataset_dir = Path("data/synthetic/test_output_20p")
    
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}")
        return
    
    variations = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("variation_")])
    
    if not variations:
        print("No variations found")
        return
    
    print("="*80)
    print("OCR Performance Analysis on Scanned PDFs")
    print("="*80)
    
    # Test with variation_01 only for speed
    variation = variations[0]
    
    for engine in ['paddle', 'tesseract']:
        print(f"\n{'='*80}")
        print(f"Testing {engine.upper()}")
        print(f"{'='*80}")
        
        result = analyze_variation(variation, engine)
        
        if result:
            print(f"\nVariation: {result['variation']}")
            print(f"Time: {result['elapsed']:.1f}s ({result['elapsed']/20:.1f}s per page)")
            print(f"Total diffs: {result['total_diffs']}")
            print(f"\nGround truth pages: {result['gt_pages']}")
            print(f"Detected pages: {result['detected_pages']}")
            print(f"\nMetrics:")
            print(f"  True Positives: {result['tp']}")
            print(f"  False Positives: {result['fp']}")
            print(f"  False Negatives: {result['fn']}")
            print(f"  Precision: {result['precision']:.1%}")
            print(f"  Recall: {result['recall']:.1%}")
            print(f"  F1 Score: {result['f1']:.1%}")


if __name__ == "__main__":
    main()
