"""Diagnostic script to trace diff bbox values during comparison."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    from extraction.pdf_parser import parse_pdf
    from comparison.text_comparison import TextComparator
    from comparison.formatting_comparison import compare_formatting
    from comparison.diff_classifier import classify_diffs
    
    path_a = project_root / "data/synthetic/dataset/variation_01/variation_01_original.pdf"
    path_b = project_root / "data/synthetic/dataset/variation_01/variation_01_modified.pdf"
    
    print(f"[INFO] Extracting {path_a.name}...")
    pages_a = parse_pdf(path_a)
    print(f"[INFO] Extracting {path_b.name}...")
    pages_b = parse_pdf(path_b)
    
    print(f"[INFO] Extracted {len(pages_a)} pages from A, {len(pages_b)} pages from B")
    
    # Print sample block bboxes
    if pages_a and pages_a[0].blocks:
        block = pages_a[0].blocks[0]
        print(f"[INFO] Sample block bbox (absolute): {block.bbox}")
        normalized = block.normalize_bbox(pages_a[0].width, pages_a[0].height)
        print(f"[INFO] Sample block bbox (normalized): {normalized}")
        print(f"[INFO] Page dimensions: {pages_a[0].width} x {pages_a[0].height}")
    
    print(f"\n[INFO] Running text comparison...")
    comparator = TextComparator()
    text_diffs = comparator.compare(pages_a, pages_b)
    
    print(f"[INFO] Found {len(text_diffs)} text diffs")
    
    # Print first 5 diff bboxes
    for i, diff in enumerate(text_diffs[:5]):
        print(f"[DIFF {i}] page={diff.page_num}, type={diff.diff_type}, change_type={diff.change_type}")
        print(f"        bbox={diff.bbox}")
        if diff.old_text:
            print(f"        old_text[:40]={diff.old_text[:40]!r}")
        if diff.new_text:
            print(f"        new_text[:40]={diff.new_text[:40]!r}")
    
    # Check formatting diffs too
    print(f"\n[INFO] Running formatting comparison...")
    formatting_diffs = compare_formatting(pages_a, pages_b)
    print(f"[INFO] Found {len(formatting_diffs)} formatting diffs")
    
    for i, diff in enumerate(formatting_diffs[:3]):
        print(f"[FORMATTING DIFF {i}] page={diff.page_num}, type={diff.diff_type}")
        print(f"        bbox={diff.bbox}")
    
    # Now test rendering
    print(f"\n[INFO] Testing render_pages with diffs...")
    all_diffs = text_diffs + formatting_diffs
    classified = classify_diffs(all_diffs)
    print(f"[INFO] Classified into {len(classified)} diffs")
    
    from visualization.pdf_viewer import render_pages
    rendered_a = render_pages(path_a, diffs=classified)
    print(f"[INFO] Rendered {len(rendered_a)} pages from A")
    
    # Save first rendered page
    if rendered_a:
        from PIL import Image
        page_num, img = rendered_a[0]
        output_path = project_root / "bbox_diag_output.png"
        Image.fromarray(img).save(output_path)
        print(f"[SUCCESS] Saved rendered page {page_num} to {output_path}")


if __name__ == "__main__":
    main()
