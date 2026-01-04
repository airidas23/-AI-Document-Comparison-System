"""Debug script to understand page 11 font size detection."""
import sys
sys.path.insert(0, '.')
from extraction import extract_pdf

# Extract pages
print("Extracting PDFs...")
pages_a = extract_pdf('data/synthetic/dataset_20p/variation_01/variation_01_original.pdf')
pages_b = extract_pdf('data/synthetic/dataset_20p/variation_01/variation_01_modified.pdf')

# Get page 11
p11_a = next((p for p in pages_a if p.page_num == 11), None)
p11_b = next((p for p in pages_b if p.page_num == 11), None)

print(f"\n=== Page 11 Original ===")
print(f"  num lines: {len(p11_a.lines) if p11_a and p11_a.lines else 0}")
print(f"  metadata: {p11_a.metadata if p11_a else None}")

print(f"\n=== Page 11 Modified ===")
print(f"  num lines: {len(p11_b.lines) if p11_b and p11_b.lines else 0}")
print(f"  metadata: {p11_b.metadata if p11_b else None}")

# Show first few lines from page 11
if p11_a and p11_a.lines:
    print("\nOriginal Page 11 first 3 lines:")
    for i, line in enumerate(p11_a.lines[:3]):
        print(f"  Line {i}: '{line.text[:60]}...' bbox={line.bbox}")

if p11_b and p11_b.lines:
    print("\nModified Page 11 first 3 lines:")
    for i, line in enumerate(p11_b.lines[:3]):
        print(f"  Line {i}: '{line.text[:60]}...' bbox={line.bbox}")
