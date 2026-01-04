#!/usr/bin/env python3
"""
Final evaluation script for PDF comparison pipeline.

Tests all variations in dataset_20p and compares detected diffs against ground truth.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.compare_pdfs import compare_pdfs


def load_change_log(variation_dir: Path) -> Dict:
    """Load change_log.json for a variation."""
    change_log_path = variation_dir / f"{variation_dir.name}_change_log.json"
    if not change_log_path.exists():
        return {}
    with open(change_log_path) as f:
        return json.load(f)


def evaluate_variation(variation_dir: Path) -> Dict:
    """Evaluate a single variation against ground truth."""
    variation_name = variation_dir.name
    original_pdf = variation_dir / f"{variation_name}_original.pdf"
    modified_pdf = variation_dir / f"{variation_name}_modified.pdf"
    
    if not original_pdf.exists() or not modified_pdf.exists():
        return {"error": "PDF files not found"}
    
    # Load ground truth
    change_log = load_change_log(variation_dir)
    ground_truth = change_log.get("changes", [])
    
    # Run comparison
    try:
        result = compare_pdfs(str(original_pdf), str(modified_pdf))
    except Exception as e:
        return {"error": str(e)}
    
    # Collect results
    diffs_by_page = defaultdict(list)
    for d in result.diffs:
        diffs_by_page[d.page_num].append({
            "change_type": d.change_type,
            "subtype": d.metadata.get("subtype", "unknown"),
            "old_text": (d.old_text[:50] + "...") if d.old_text and len(d.old_text) > 50 else d.old_text,
            "new_text": (d.new_text[:50] + "...") if d.new_text and len(d.new_text) > 50 else d.new_text,
        })
    
    # Extract ground truth pages
    gt_pages = set(c.get("page", 0) for c in ground_truth)
    detected_pages = set(diffs_by_page.keys())
    
    return {
        "variation": variation_name,
        "ground_truth_count": len(ground_truth),
        "ground_truth_pages": sorted(gt_pages),
        "detected_count": len(result.diffs),
        "detected_pages": sorted(detected_pages),
        "true_positives": len(gt_pages & detected_pages),
        "false_positives": len(detected_pages - gt_pages),
        "false_negatives": len(gt_pages - detected_pages),
        "diffs_by_page": dict(diffs_by_page),
        "ground_truth": ground_truth,
    }


def main():
    dataset_dir = Path("data/synthetic/dataset_20p")
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return 1
    
    variations = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("variation_")])
    
    if not variations:
        print("No variations found")
        return 1
    
    print("=" * 80)
    print("PDF Comparison Pipeline - Final Evaluation")
    print("=" * 80)
    print()
    
    total_gt = 0
    total_detected = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    results = []
    
    for variation_dir in variations:
        print(f"\n{'='*60}")
        print(f"Evaluating: {variation_dir.name}")
        print(f"{'='*60}")
        
        result = evaluate_variation(variation_dir)
        results.append(result)
        
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        
        print(f"  Ground Truth: {result['ground_truth_count']} changes on pages {result['ground_truth_pages']}")
        print(f"  Detected: {result['detected_count']} diffs on pages {result['detected_pages']}")
        print(f"  True Positives (page overlap): {result['true_positives']}")
        print(f"  False Positives: {result['false_positives']}")
        print(f"  False Negatives: {result['false_negatives']}")
        
        total_gt += result['ground_truth_count']
        total_detected += result['detected_count']
        total_tp += result['true_positives']
        total_fp += result['false_positives']
        total_fn += result['false_negatives']
        
        # Show diffs by page
        print("\n  Diffs by page:")
        for page in sorted(result['diffs_by_page'].keys()):
            diffs = result['diffs_by_page'][page]
            print(f"    Page {page}: {len(diffs)} diff(s)")
            for d in diffs[:3]:  # Show first 3
                print(f"      - {d['change_type']}:{d['subtype']}")
            if len(diffs) > 3:
                print(f"      ... and {len(diffs) - 3} more")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total variations evaluated: {len(results)}")
    print(f"Total ground truth changes: {total_gt}")
    print(f"Total detected diffs: {total_detected}")
    print(f"Total true positives (page-level): {total_tp}")
    print(f"Total false positives (extra pages): {total_fp}")
    print(f"Total false negatives (missed pages): {total_fn}")
    
    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
        print(f"\nPage-level Precision: {precision:.2%}")
    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
        print(f"Page-level Recall: {recall:.2%}")
    if total_tp + total_fp + total_fn > 0:
        f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
        print(f"Page-level F1 Score: {f1:.2%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
