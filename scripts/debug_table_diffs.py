#!/usr/bin/env python3
"""Debug table:structure diffs to understand false positives."""

from pipeline.compare_pdfs import compare_pdfs
from pathlib import Path

base = Path('data/synthetic/dataset_20p')
original = base / 'variation_01' / 'variation_01_original.pdf'
modified = base / 'variation_01' / 'variation_01_modified.pdf'

result = compare_pdfs(str(original), str(modified))

# Get all table:structure diffs
print("All diffs:")
print("=" * 80)
for diff in result.diffs:
    print(f'Page {diff.page_num}: {diff.change_type}:{diff.metadata.get("subtype", "unknown")}')
    if 'table' in diff.metadata.get("subtype", ""):
        print(f'  Old text: {repr(diff.old_text[:150]) if diff.old_text else None}')
        print(f'  New text: {repr(diff.new_text[:150]) if diff.new_text else None}')
    print()
