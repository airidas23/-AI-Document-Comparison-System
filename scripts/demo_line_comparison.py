"""Demo LINE-level comparison on synthetic PDFs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparison.alignment import align_pages
from comparison.diff_classifier import classify_diffs, get_diff_summary
from comparison.line_comparison import compare_lines
from comparison.models import ComparisonResult
from export.json_exporter import export_json
from extraction import extract_lines


def _load_change_log(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("changes", [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variation", default="variation_01")
    parser.add_argument("--dataset-root", default="data/synthetic/dataset")
    parser.add_argument("--output", default="line_comparison_output.json")
    args = parser.parse_args()

    variation_dir = Path(args.dataset_root) / args.variation
    original_pdf = variation_dir / f"{args.variation}_original.pdf"
    modified_pdf = variation_dir / f"{args.variation}_modified.pdf"
    change_log = variation_dir / f"{args.variation}_change_log.json"

    pages_a = extract_lines(original_pdf)
    pages_b = extract_lines(modified_pdf)

    alignment_map = align_pages(pages_a, pages_b, use_similarity=False)
    diffs = compare_lines(pages_a, pages_b, alignment_map=alignment_map)
    classified = classify_diffs(diffs)

    result = ComparisonResult(
        doc1=str(original_pdf),
        doc2=str(modified_pdf),
        pages=pages_a,
        diffs=classified,
        summary=get_diff_summary(classified),
    )

    output_path = Path(args.output)
    export_json(result, output_path)

    detected_counts = Counter(d.change_type for d in classified)
    expected_counts = Counter()
    if change_log.exists():
        for change in _load_change_log(change_log):
            expected_counts[change.get("change_type", "unknown")] += 1

    print(f"output: {output_path}")
    print(f"diffs detected: {len(classified)}")
    print(f"detected by change_type: {dict(detected_counts)}")
    if expected_counts:
        print(f"expected by change_type: {dict(expected_counts)}")


if __name__ == "__main__":
    main()
