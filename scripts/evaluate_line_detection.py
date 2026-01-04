"""Evaluate LINE-level diffs against synthetic ground truth."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparison.alignment import align_pages
from comparison.diff_classifier import classify_diffs
from comparison.line_comparison import compare_lines
from utils.text_normalization import normalize_text
from extraction import extract_lines


TARGET_TYPES = [
    "spacing_change",
    "table_column_addition",
    "caption_number_change",
    "typo",
    "comma_removal",
]


def _load_change_log(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("changes", [])


def _strip_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]+", "", text)


def _is_punctuation_only_change(text_a: str, text_b: str) -> bool:
    return normalize_text(_strip_punctuation(text_a)) == normalize_text(_strip_punctuation(text_b)) and text_a != text_b


def _extract_figure_caption_suffix(text: str) -> tuple[str, str] | None:
    match = re.match(r"^(figure|fig\.)\s*\d+\.?\s*(.*)$", text.strip(), re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower(), match.group(2).strip()


def _categorize_expected(change: dict) -> str | None:
    desc = (change.get("description") or "").lower()
    region = (change.get("region") or "").lower()

    if "spacing" in desc or change.get("change_type") == "layout":
        return "spacing_change"
    if "delta column" in desc or (region == "table" and "column" in desc):
        return "table_column_addition"
    if "caption" in desc or "figure" in desc:
        return "caption_number_change"
    if "typo" in desc or "character duplication" in desc:
        return "typo"
    if "comma" in desc or "punctuation" in desc:
        return "comma_removal"
    return None


def _categorize_detected(diff) -> list[str]:
    categories: list[str] = []
    old_text = diff.old_text or ""
    new_text = diff.new_text or ""

    if diff.change_type == "layout":
        categories.append("spacing_change")

    if new_text and "delta" in new_text.lower() and "delta" not in old_text.lower():
        categories.append("table_column_addition")

    old_caption = _extract_figure_caption_suffix(old_text)
    new_caption = _extract_figure_caption_suffix(new_text)
    if old_caption and new_caption and old_caption[0] == new_caption[0]:
        if old_caption[1] == new_caption[1]:
            categories.append("caption_number_change")

    if diff.change_type == "content" and old_text and new_text:
        similarity = fuzz.ratio(normalize_text(old_text), normalize_text(new_text)) / 100.0
        if similarity >= 0.92 and not _is_punctuation_only_change(old_text, new_text):
            categories.append("typo")

    if diff.change_type == "formatting" and old_text and new_text:
        if _is_punctuation_only_change(old_text, new_text):
            categories.append("comma_removal")

    return categories


def _precision_recall(matches: int, detected: int, expected: int) -> tuple[float, float]:
    precision = matches / detected if detected else 0.0
    recall = matches / expected if expected else 0.0
    return precision, recall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variation", default="variation_01")
    parser.add_argument("--dataset-root", default="data/synthetic/dataset")
    args = parser.parse_args()

    variation_dir = Path(args.dataset_root) / args.variation
    original_pdf = variation_dir / f"{args.variation}_original.pdf"
    modified_pdf = variation_dir / f"{args.variation}_modified.pdf"
    change_log = variation_dir / f"{args.variation}_change_log.json"

    expected_counts = {key: 0 for key in TARGET_TYPES}
    for change in _load_change_log(change_log):
        category = _categorize_expected(change)
        if category in expected_counts:
            expected_counts[category] += 1

    pages_a = extract_lines(original_pdf)
    pages_b = extract_lines(modified_pdf)
    alignment_map = align_pages(pages_a, pages_b, use_similarity=False)
    diffs = classify_diffs(compare_lines(pages_a, pages_b, alignment_map=alignment_map))

    detected_by_type = {key: [] for key in TARGET_TYPES}
    for diff in diffs:
        for category in _categorize_detected(diff):
            if category in detected_by_type:
                detected_by_type[category].append(diff)

    print(f"Evaluation: {args.variation}")
    for category in TARGET_TYPES:
        detected = len(detected_by_type[category])
        expected = expected_counts.get(category, 0)
        matches = min(detected, expected)
        precision, recall = _precision_recall(matches, detected, expected)
        print(
            f"{category}: detected={detected} expected={expected} "
            f"precision={precision:.2f} recall={recall:.2f}"
        )


if __name__ == "__main__":
    main()
