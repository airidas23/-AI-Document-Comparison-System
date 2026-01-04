"""Evaluate pipeline diffs against synthetic change logs.

Usage:
    python scripts/evaluate_variation.py --pair variation_01
    python scripts/evaluate_variation.py --log data/synthetic/dataset/variation_01/variation_01_change_log.json

    # Evaluate scanned PDFs referenced by the change log (if present)
    python scripts/evaluate_variation.py --scanned --log data/synthetic/test_scanned_dataset/variation_01/variation_01_change_log.json

Exit code:
    0 if every logged change is matched by at least one detected diff, else 1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.compare_pdfs import compare_pdfs
from utils.text_normalization import normalize_text


_NUM_TOKEN_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class ExpectedChange:
    page: int
    region: str
    diff_type: str
    change_type: str
    description: str
    before: Optional[str]
    after: Optional[str]


def _load_change_log(path: Path) -> Tuple[Dict[str, Any], List[ExpectedChange]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changes: List[ExpectedChange] = []
    for item in payload.get("changes", []):
        changes.append(
            ExpectedChange(
                page=int(item.get("page", 1)),
                region=str(item.get("region", "")),
                diff_type=str(item.get("diff_type", "")),
                change_type=str(item.get("change_type", "")),
                description=str(item.get("description", "")),
                before=item.get("before"),
                after=item.get("after"),
            )
        )
    return payload, changes


def _normalize_for_match(text: Optional[str]) -> str:
    return normalize_text(text or "")


def _strip_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]+", "", text)


def _contains_delta_column_change(diff: Dict[str, Any]) -> bool:
    # Table column add often appears as a structure diff OR multiple cell diffs.
    meta = diff.get("metadata") or {}
    table_change = meta.get("table_change")
    if table_change == "structure":
        old_cols = (meta.get("old_structure") or {}).get("cols")
        new_cols = (meta.get("new_structure") or {}).get("cols")
        try:
            return old_cols is not None and new_cols is not None and int(new_cols) > int(old_cols)
        except Exception:
            return False

    # Heuristic: a cell content diff that introduces an extra numeric token.
    old_text = _normalize_for_match(diff.get("old_text"))
    new_text = _normalize_for_match(diff.get("new_text"))
    if not old_text or not new_text:
        return False

    old_tokens = old_text.split()
    new_tokens = new_text.split()
    old_nums = sum(1 for t in old_tokens if _NUM_TOKEN_RE.match(t))
    new_nums = sum(1 for t in new_tokens if _NUM_TOKEN_RE.match(t))
    return new_nums > old_nums


def _diffs_for_page(diffs: List[Dict[str, Any]], page: int) -> List[Dict[str, Any]]:
    return [d for d in diffs if int(d.get("page", d.get("page_num", 0)) or 0) == page]


def _best_match_for_change(
    change: ExpectedChange,
    diffs: List[Dict[str, Any]],
    available_pages: int,
) -> Optional[Dict[str, Any]]:
    # Reconcile potentially-wrong logged pages: clamp into actual page range.
    page = min(max(change.page, 1), max(available_pages, 1))
    page_diffs = _diffs_for_page(diffs, page)

    # Layout-only expected changes may not include before/after text.
    if change.change_type == "layout" and not change.before and not change.after:
        for d in page_diffs:
            if d.get("change_type") == "layout":
                meta = d.get("metadata") or {}
                if meta.get("subtype") == "layout_shift":
                    return d
        # Fallback: any layout diff on the page.
        for d in page_diffs:
            if d.get("change_type") == "layout":
                return d

    # Region-specific matching.
    if change.region == "table" and "delta column" in change.description.lower():
        for d in page_diffs:
            if d.get("type") == "table" and _contains_delta_column_change(d):
                return d
        # Fallback: any diff that looks like delta column addition.
        for d in page_diffs:
            if _contains_delta_column_change(d):
                return d
        return None

    if change.region == "figure_caption":
        for d in page_diffs:
            meta = d.get("metadata") or {}
            if d.get("type") == "figure" and meta.get("figure_change") in {"numbering", "caption_text"}:
                return d
        return None

    # Text changes: try to match by before/after substring or normalized equality.
    before = _normalize_for_match(change.before)
    after = _normalize_for_match(change.after)

    candidates = [d for d in page_diffs if d.get("type") == "text" and d.get("diff_type") == change.diff_type]
    if not candidates:
        candidates = [d for d in page_diffs if d.get("diff_type") == change.diff_type]

    def score(d: Dict[str, Any]) -> float:
        d_before = _normalize_for_match(d.get("text_before") or d.get("old_text"))
        d_after = _normalize_for_match(d.get("text_after") or d.get("new_text"))

        if before and before == d_before:
            s = 2.0
        elif before and before in d_before:
            s = 1.0
        else:
            s = 0.0

        if after and after == d_after:
            s += 2.0
        elif after and after in d_after:
            s += 1.0

        # Allow punctuation-only changes to match even if classifier differs.
        if change.change_type == "formatting" and before and after:
            if normalize_text(_strip_punctuation(before)) == normalize_text(_strip_punctuation(after)):
                if normalize_text(_strip_punctuation(d_before)) == normalize_text(_strip_punctuation(d_after)):
                    s += 0.5

        # Prefer same change_type when available.
        if change.change_type and d.get("change_type") == change.change_type:
            s += 0.25

        return s

    best = None
    best_s = 0.0
    for d in candidates:
        s = score(d)
        if s > best_s:
            best_s = s
            best = d

    return best if best_s >= 1.0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", help="variation id, e.g. variation_01")
    parser.add_argument("--log", help="path to variation_*_change_log.json")
    parser.add_argument(
        "--scanned",
        action="store_true",
        help="Use original_scanned_pdf/modified_scanned_pdf from the change log (if present)",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/synthetic/dataset",
        help="dataset root containing variation folders",
    )
    args = parser.parse_args()

    if not args.pair and not args.log:
        raise SystemExit("Provide --pair or --log")

    if args.log:
        log_path = Path(args.log)
    else:
        log_path = Path(args.dataset_root) / args.pair / f"{args.pair}_change_log.json"

    payload, expected = _load_change_log(log_path)

    def _pick_pdf(key: str, fallback_key: str) -> Path:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return Path(val)
        return Path(payload[fallback_key])

    if args.scanned:
        original_pdf = _pick_pdf("original_scanned_pdf", "original_pdf")
        modified_pdf = _pick_pdf("modified_scanned_pdf", "modified_pdf")
    else:
        original_pdf = Path(payload["original_pdf"])
        modified_pdf = Path(payload["modified_pdf"])

    result = compare_pdfs(str(original_pdf), str(modified_pdf))

    # The pipeline returns Diff dataclasses; normalize into dicts for scoring.
    diffs: List[Dict[str, Any]] = []
    for d in result.diffs:
        dd = {
            "type": d.metadata.get("type"),
            "page": d.page_num,
            "diff_type": d.diff_type,
            "change_type": d.change_type,
            "old_text": d.old_text,
            "new_text": d.new_text,
            "metadata": d.metadata,
        }
        # Back-compat with older display fields
        dd["text_before"] = d.old_text
        dd["text_after"] = d.new_text
        diffs.append(dd)

    available_pages = max((p.page_num for p in result.pages), default=1)

    matched: List[Tuple[ExpectedChange, Dict[str, Any]]] = []
    unmatched: List[ExpectedChange] = []

    for change in expected:
        m = _best_match_for_change(change, diffs, available_pages)
        if m is None:
            unmatched.append(change)
        else:
            matched.append((change, m))

    print(f"log: {log_path}")
    print(f"pdfs: {original_pdf.name} vs {modified_pdf.name}")
    print(f"expected changes: {len(expected)}")
    print(f"detected diffs: {len(result.diffs)}")
    print(f"matched: {len(matched)}")
    print(f"unmatched: {len(unmatched)}")

    if unmatched:
        print("\nUnmatched expected changes:")
        for c in unmatched:
            print(f"- page={c.page} region={c.region} type={c.diff_type}/{c.change_type}: {c.description}")

    return 0 if not unmatched else 1


if __name__ == "__main__":
    raise SystemExit(main())
