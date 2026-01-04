"""Generate a detailed mismatch report for the synthetic golden dataset.

The existing golden evaluation export (tests/golden_results.json) contains only
aggregate metrics. This script re-runs the pipeline for each variation and writes
per-variation TP/FP/FN examples to debug_output/golden_mismatch_report.json.

Usage:
  .venv/bin/python scripts/golden_mismatch_report.py

Output:
  debug_output/golden_mismatch_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import compare_pdfs
SYNTHETIC_DATASET_DIR = PROJECT_ROOT / "data" / "synthetic" / "dataset"
OUT_PATH = PROJECT_ROOT / "debug_output" / "golden_mismatch_report.json"

# Reuse GT loader from tests (keeps GT parsing consistent with evaluation)
from tests.test_golden_evaluation import load_ground_truth


def _norm(text: str | None) -> str:
    return (text or "").strip().lower()


def _text_similarity(a: str | None, b: str | None) -> float:
    a_n = _norm(a)
    b_n = _norm(b)
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0

    try:
        from rapidfuzz.fuzz import partial_ratio

        return partial_ratio(a_n, b_n) / 100.0
    except Exception:
        if a_n in b_n or b_n in a_n:
            return 1.0
        ta = set(a_n.split())
        tb = set(b_n.split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)


def _pair_score(pred, gt) -> float:
    # Page must match
    if pred.page_num != gt.page:
        return -1.0

    # Diff type compatibility (allow modified with anything)
    if pred.diff_type != gt.diff_type and not (pred.diff_type == "modified" or gt.diff_type == "modified"):
        return -1.0

    gt_before = _norm(gt.before)
    gt_after = _norm(gt.after)
    gt_has_text = bool(gt_before or gt_after)

    # If GT has no text (layout-ish), rely on change_type
    if not gt_has_text:
        return 1.0 if pred.change_type == gt.change_type else 0.4

    candidates = [
        _text_similarity(pred.old_text, gt.before),
        _text_similarity(pred.new_text, gt.after),
        _text_similarity(pred.old_text, gt.after),
        _text_similarity(pred.new_text, gt.before),
    ]
    best = max(candidates)

    # Small bonus if change_type matches
    if pred.change_type == gt.change_type:
        best = min(1.0, best + 0.05)

    return best


def _summarize_pred(diff) -> dict:
    return {
        "page": diff.page_num,
        "diff_type": str(diff.diff_type),
        "change_type": str(diff.change_type),
        "old_text": (diff.old_text or "")[:160],
        "new_text": (diff.new_text or "")[:160],
        "bbox": diff.bbox,
        "confidence": diff.confidence,
        "metadata": {
            k: v
            for k, v in (diff.metadata or {}).items()
            if k in {"region", "severity", "description", "source", "rule", "kind"}
        },
    }


def _summarize_gt(gt) -> dict:
    return {
        "page": gt.page,
        "diff_type": str(gt.diff_type),
        "change_type": str(gt.change_type),
        "before": (gt.before or "")[:160],
        "after": (gt.after or "")[:160],
        "region": gt.region,
        "severity": gt.severity,
        "description": gt.description,
    }


def main() -> int:
    if not SYNTHETIC_DATASET_DIR.exists():
        raise SystemExit(f"Synthetic dataset not found: {SYNTHETIC_DATASET_DIR}")

    variations = sorted(
        [p for p in SYNTHETIC_DATASET_DIR.iterdir() if p.is_dir() and p.name.startswith("variation_")]
    )

    report: dict = {
        "dataset_dir": str(SYNTHETIC_DATASET_DIR),
        "variations": [],
    }

    for variation_dir in variations:
        pair_id = variation_dir.name
        change_log_path = variation_dir / f"{pair_id}_change_log.json"
        original_pdf = variation_dir / f"{pair_id}_original.pdf"
        modified_pdf = variation_dir / f"{pair_id}_modified.pdf"

        if not (change_log_path.exists() and original_pdf.exists() and modified_pdf.exists()):
            continue

        gt_changes = load_ground_truth(change_log_path)
        result = compare_pdfs(original_pdf, modified_pdf)
        preds = list(result.diffs)

        # Build candidate edges with scores
        edges: list[tuple[float, int, int]] = []
        for pi, pred in enumerate(preds):
            for gi, gt in enumerate(gt_changes):
                score = _pair_score(pred, gt)
                if score < 0:
                    continue

                gt_has_text = bool(_norm(gt.before) or _norm(gt.after))
                if (gt_has_text and score >= 0.60) or (not gt_has_text and score >= 0.40):
                    edges.append((score, pi, gi))

        # Greedy max-score matching (one-to-one)
        edges.sort(reverse=True)
        matched_pred: set[int] = set()
        matched_gt: set[int] = set()
        matches: list[dict] = []

        for score, pi, gi in edges:
            if pi in matched_pred or gi in matched_gt:
                continue
            matched_pred.add(pi)
            matched_gt.add(gi)
            matches.append({
                "score": round(float(score), 3),
                "pred": _summarize_pred(preds[pi]),
                "gt": _summarize_gt(gt_changes[gi]),
            })

        fps = [_summarize_pred(preds[i]) for i in range(len(preds)) if i not in matched_pred]
        fns = [_summarize_gt(gt_changes[i]) for i in range(len(gt_changes)) if i not in matched_gt]

        # Near-misses for FNs: best candidate score even if below threshold
        near_misses: list[dict] = []
        for gi, gt in enumerate(gt_changes):
            if gi in matched_gt:
                continue
            best_score = -1.0
            best_pi: int | None = None
            for pi, pred in enumerate(preds):
                score = _pair_score(pred, gt)
                if score > best_score:
                    best_score = score
                    best_pi = pi
            if best_pi is not None and best_score >= 0:
                near_misses.append({
                    "best_score": round(float(best_score), 3),
                    "gt": _summarize_gt(gt),
                    "best_pred": _summarize_pred(preds[best_pi]),
                })

        near_misses.sort(key=lambda x: x["best_score"], reverse=True)

        report["variations"].append({
            "pair_id": pair_id,
            "gt_count": len(gt_changes),
            "pred_count": len(preds),
            "tp": len(matches),
            "fp": len(fps),
            "fn": len(fns),
            "matches": matches,
            "false_positives": fps,
            "false_negatives": fns,
            "near_misses": near_misses[:5],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote mismatch report: {OUT_PATH}")
    for v in report["variations"]:
        print(f"{v['pair_id']}: TP={v['tp']} FP={v['fp']} FN={v['fn']} (GT={v['gt_count']} Pred={v['pred_count']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
