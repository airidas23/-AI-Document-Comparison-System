"""
Golden evaluation tests using synthetic dataset with ground truth.

This module tests the comparison pipeline against known ground truth
from the synthetic dataset generator, calculating precision, recall, and F1 score.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional
import difflib

import pytest

from comparison.models import ChangeType, Diff, DiffType
from utils.metrics import (
    ChangeDetectionMetrics,
    calculate_change_detection_metrics,
    calculate_change_type_metrics,
    calculate_performance_metrics,
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SYNTHETIC_DATASET_DIR = PROJECT_ROOT / "data" / "synthetic" / "dataset"
GOLDEN_RESULTS_PATH = PROJECT_ROOT / "tests" / "golden_results.json"
MISMATCH_REPORT_PATH = PROJECT_ROOT / "debug_output" / "golden_mismatch_report.json"


@dataclass
class GroundTruthChange:
    """A single ground truth change from the synthetic dataset."""
    page: int
    region: str
    diff_type: DiffType
    change_type: ChangeType
    severity: str
    description: str
    before: Optional[str] = None
    after: Optional[str] = None
    
    def to_diff(self) -> Diff:
        """Convert to Diff object for comparison."""
        return Diff(
            page_num=self.page,
            diff_type=self.diff_type,
            change_type=self.change_type,
            old_text=self.before,
            new_text=self.after,
            bbox=None,  # Ground truth doesn't have precise bbox
            confidence=1.0,
            metadata={
                "region": self.region,
                "severity": self.severity,
                "description": self.description,
                "source": "ground_truth",
            },
        )


@dataclass
class VariationResult:
    """Result from evaluating a single variation."""
    pair_id: str
    metrics: ChangeDetectionMetrics
    metrics_by_type: Dict[str, ChangeDetectionMetrics]
    performance: Dict[str, float]
    ground_truth_count: int
    predicted_count: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pair_id": self.pair_id,
            "metrics": self.metrics.to_dict(),
            "metrics_by_type": {k: v.to_dict() for k, v in self.metrics_by_type.items()},
            "performance": self.performance,
            "ground_truth_count": self.ground_truth_count,
            "predicted_count": self.predicted_count,
        }


def load_ground_truth(change_log_path: Path) -> List[GroundTruthChange]:
    """
    Load ground truth changes from a change log JSON file.
    
    Args:
        change_log_path: Path to variation_*_change_log.json
    
    Returns:
        List of GroundTruthChange objects
    """
    with open(change_log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    changes = []
    for change in data.get("changes", []):
        # Map change_type from ground truth to our ChangeType
        change_type_raw = change.get("change_type", "content")
        if change_type_raw not in ("content", "formatting", "layout", "visual"):
            change_type_raw = "content"
        
        # Map diff_type
        diff_type_raw = change.get("diff_type", "modified")
        if diff_type_raw not in ("added", "deleted", "modified"):
            diff_type_raw = "modified"
        
        gt_change = GroundTruthChange(
            page=change.get("page", 1),
            region=change.get("region", "unknown"),
            diff_type=diff_type_raw,  # type: ignore
            change_type=change_type_raw,  # type: ignore
            severity=change.get("severity", "minor"),
            description=change.get("description", ""),
            before=change.get("before"),
            after=change.get("after"),
        )
        changes.append(gt_change)
    
    return changes


def match_diff_to_ground_truth(
    pred_diff: Diff,
    gt_changes: List[GroundTruthChange],
    tolerance: float = 0.6,
) -> Optional[GroundTruthChange]:
    """
    Match a predicted diff to ground truth using relaxed matching.
    
    Matching criteria:
    1. Same page
    2. Same or compatible diff_type
    3. Text similarity (if available)
    """
    def _ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _text_match_score(gt_before: str, gt_after: str, pred_old: str, pred_new: str) -> float:
        # Quick substring checks first (fast, robust when one side contains the other)
        needle_len = 30
        pairs = [
            (gt_before, pred_old),
            (gt_after, pred_new),
            (gt_before, pred_new),
            (gt_after, pred_old),
        ]
        for a, b in pairs:
            if not a or not b:
                continue
            a_sub = a[:needle_len]
            b_sub = b[:needle_len]
            if a_sub and (a_sub in b):
                return 1.0
            if b_sub and (b_sub in a):
                return 1.0

        # Fuzzy match: take the best similarity among plausible field pairings.
        ratios = [
            _ratio(gt_before, pred_old),
            _ratio(gt_after, pred_new),
            _ratio(gt_before, pred_new),
            _ratio(gt_after, pred_old),
        ]
        return max(ratios) if ratios else 0.0

    best_gt: Optional[GroundTruthChange] = None
    best_score = 0.0

    for gt in gt_changes:
        # Must be same page
        if pred_diff.page_num != gt.page:
            continue

        # Header/footer diffs are very specific; prevent them from matching unrelated
        # formatting-only GT entries (e.g., font size) that often have no before/after.
        pred_hf_kind = (pred_diff.metadata or {}).get("header_footer_change")
        gt_is_hf = gt.region in {"header", "footer", "header_footer"} or gt.description.lower().startswith(
            "updated header"
        ) or gt.description.lower().startswith("updated footer")
        if pred_hf_kind in {"header", "footer", "header_footer"} and not gt_is_hf:
            continue
        if gt_is_hf and pred_hf_kind not in {"header", "footer", "header_footer"}:
            continue

        # Check diff_type compatibility
        if pred_diff.diff_type != gt.diff_type:
            # Allow some flexibility: "modified" can match others
            if not (pred_diff.diff_type == "modified" or gt.diff_type == "modified"):
                continue

        pred_old = (pred_diff.old_text or "").strip().lower()
        pred_new = (pred_diff.new_text or "").strip().lower()
        gt_before = (gt.before or "").strip().lower()
        gt_after = (gt.after or "").strip().lower()

        # If ground-truth contains text, require a minimum similarity.
        if gt_before or gt_after:
            score = _text_match_score(gt_before, gt_after, pred_old, pred_new)
            if score >= tolerance and score > best_score:
                best_score = score
                best_gt = gt
            continue

        # Ground truth has no text (e.g., layout change): match based on page + change_type.
        if pred_diff.change_type == gt.change_type:
            return gt
        # Only allow layout<->formatting compatibility for formatting diffs that *look*
        # like layout-only changes (no text payload).
        if gt.change_type == "layout" and pred_diff.change_type == "formatting":
            if (pred_diff.old_text is None and pred_diff.new_text is None) or (
                (pred_diff.metadata or {}).get("subtype") in {"layout_drift", "layout_shift", "layout_translation"}
            ):
                return gt

    return best_gt


def match_diff_to_ground_truth_with_score(
    pred_diff: Diff,
    gt_changes: List[GroundTruthChange],
    tolerance: float = 0.6,
) -> tuple[Optional[GroundTruthChange], float]:
    """Like match_diff_to_ground_truth, but also returns the best match score."""

    def _ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _text_match_score(gt_before: str, gt_after: str, pred_old: str, pred_new: str) -> float:
        needle_len = 30
        pairs = [
            (gt_before, pred_old),
            (gt_after, pred_new),
            (gt_before, pred_new),
            (gt_after, pred_old),
        ]
        for a, b in pairs:
            if not a or not b:
                continue
            a_sub = a[:needle_len]
            b_sub = b[:needle_len]
            if a_sub and (a_sub in b):
                return 1.0
            if b_sub and (b_sub in a):
                return 1.0

        ratios = [
            _ratio(gt_before, pred_old),
            _ratio(gt_after, pred_new),
            _ratio(gt_before, pred_new),
            _ratio(gt_after, pred_old),
        ]
        return max(ratios) if ratios else 0.0

    best_gt: Optional[GroundTruthChange] = None
    best_score = 0.0

    for gt in gt_changes:
        if pred_diff.page_num != gt.page:
            continue

        pred_hf_kind = (pred_diff.metadata or {}).get("header_footer_change")
        gt_is_hf = gt.region in {"header", "footer", "header_footer"} or gt.description.lower().startswith(
            "updated header"
        ) or gt.description.lower().startswith("updated footer")
        if pred_hf_kind in {"header", "footer", "header_footer"} and not gt_is_hf:
            continue
        if gt_is_hf and pred_hf_kind not in {"header", "footer", "header_footer"}:
            continue

        if pred_diff.diff_type != gt.diff_type:
            if not (pred_diff.diff_type == "modified" or gt.diff_type == "modified"):
                continue

        pred_old = (pred_diff.old_text or "").strip().lower()
        pred_new = (pred_diff.new_text or "").strip().lower()
        gt_before = (gt.before or "").strip().lower()
        gt_after = (gt.after or "").strip().lower()

        if gt_before or gt_after:
            score = _text_match_score(gt_before, gt_after, pred_old, pred_new)
            if score >= tolerance and score > best_score:
                best_score = score
                best_gt = gt
            continue

        if pred_diff.change_type == gt.change_type:
            return gt, 1.0
        if gt.change_type == "layout" and pred_diff.change_type == "formatting":
            if (pred_diff.old_text is None and pred_diff.new_text is None) or (
                (pred_diff.metadata or {}).get("subtype") in {"layout_drift", "layout_shift", "layout_translation"}
            ):
                return gt, 1.0

    return best_gt, best_score


def calculate_soft_metrics(
    predicted_diffs: List[Diff],
    ground_truth_changes: List[GroundTruthChange],
) -> ChangeDetectionMetrics:
    """
    Calculate metrics with soft matching (text overlap, same page).
    
    This is more appropriate for our synthetic dataset where ground truth
    doesn't have precise bbox coordinates.
    """
    if not ground_truth_changes:
        if not predicted_diffs:
            return ChangeDetectionMetrics(precision=1.0, recall=1.0, f1_score=1.0, accuracy=1.0)
        return ChangeDetectionMetrics(precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0)
    
    if not predicted_diffs:
        return ChangeDetectionMetrics(precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0)
    
    # Track matches (one-to-one): each ground-truth change can be matched by at most one prediction.
    unmatched_gt = list(ground_truth_changes)
    matched_pred_indices = set()
    matched_gt_indices = set()

    def _match_priority(diff: Diff) -> tuple:
        # Prefer matching layout-only (no-text) diffs first so they don't get
        # "stolen" by unrelated formatting changes.
        has_text = bool((diff.old_text or "").strip() or (diff.new_text or "").strip())
        layout_only = (diff.change_type == "layout") and (not has_text)
        return (
            0 if layout_only else 1,
            0 if diff.change_type == "layout" else 1,
            0 if not has_text else 1,
        )

    for pred_idx, pred_diff in sorted(list(enumerate(predicted_diffs)), key=lambda x: _match_priority(x[1])):
        matched_gt = match_diff_to_ground_truth(pred_diff, unmatched_gt)
        if not matched_gt:
            continue
        # Resolve indices against the original GT list for metrics bookkeeping.
        gt_idx = ground_truth_changes.index(matched_gt)
        matched_pred_indices.add(pred_idx)
        matched_gt_indices.add(gt_idx)
        # Remove from unmatched pool to keep matching one-to-one.
        unmatched_gt.remove(matched_gt)

    tp = len(matched_gt_indices)
    fp = len(predicted_diffs) - len(matched_pred_indices)
    fn = len(ground_truth_changes) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    return ChangeDetectionMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        accuracy=accuracy,
    )


def _diff_to_report_dict(diff: Diff) -> dict:
    return {
        "page": int(diff.page_num),
        "diff_type": str(diff.diff_type),
        "change_type": str(diff.change_type),
        "old_text": diff.old_text,
        "new_text": diff.new_text,
        "bbox": diff.bbox,
        "confidence": float(diff.confidence) if diff.confidence is not None else None,
        "metadata": diff.metadata or {},
    }


def _gt_to_report_dict(gt: GroundTruthChange) -> dict:
    return {
        "page": int(gt.page),
        "diff_type": str(gt.diff_type),
        "change_type": str(gt.change_type),
        "before": gt.before,
        "after": gt.after,
        "region": gt.region,
        "severity": gt.severity,
        "description": gt.description,
    }


def evaluate_variation(variation_dir: Path) -> Optional[VariationResult]:
    """
    Evaluate a single variation against its ground truth.
    
    Args:
        variation_dir: Path to variation directory (e.g., variation_01/)
    
    Returns:
        VariationResult or None if files are missing
    """
    from pipeline import compare_pdfs
    
    pair_id = variation_dir.name
    change_log_path = variation_dir / f"{pair_id}_change_log.json"
    original_pdf = variation_dir / f"{pair_id}_original.pdf"
    modified_pdf = variation_dir / f"{pair_id}_modified.pdf"
    
    # Check if all required files exist
    if not all(p.exists() for p in [change_log_path, original_pdf, modified_pdf]):
        return None
    
    # Load ground truth
    ground_truth_changes = load_ground_truth(change_log_path)
    
    # Run comparison pipeline
    start_time = time.time()
    result = compare_pdfs(original_pdf, modified_pdf)
    total_time = time.time() - start_time
    
    predicted_diffs = result.diffs
    
    # Calculate metrics with soft matching
    metrics = calculate_soft_metrics(predicted_diffs, ground_truth_changes)
    
    # Calculate per-type metrics
    gt_diffs = [gt.to_diff() for gt in ground_truth_changes]
    metrics_by_type = calculate_change_type_metrics(predicted_diffs, gt_diffs)
    
    # Performance metrics
    performance = {
        "total_time_seconds": total_time,
        "pages_processed": len(result.pages),
        "time_per_page": total_time / max(1, len(result.pages)),
    }
    
    return VariationResult(
        pair_id=pair_id,
        metrics=metrics,
        metrics_by_type=metrics_by_type,
        performance=performance,
        ground_truth_count=len(ground_truth_changes),
        predicted_count=len(predicted_diffs),
    )


def evaluate_variation_with_debug(variation_dir: Path) -> Optional[tuple[VariationResult, dict]]:
    """Evaluate a variation and also return a mismatch report item (single pipeline run)."""
    from pipeline import compare_pdfs

    pair_id = variation_dir.name
    change_log_path = variation_dir / f"{pair_id}_change_log.json"
    original_pdf = variation_dir / f"{pair_id}_original.pdf"
    modified_pdf = variation_dir / f"{pair_id}_modified.pdf"

    if not all(p.exists() for p in [change_log_path, original_pdf, modified_pdf]):
        return None

    ground_truth_changes = load_ground_truth(change_log_path)

    start_time = time.time()
    result = compare_pdfs(original_pdf, modified_pdf)
    total_time = time.time() - start_time

    predicted_diffs = result.diffs
    metrics = calculate_soft_metrics(predicted_diffs, ground_truth_changes)

    gt_diffs = [gt.to_diff() for gt in ground_truth_changes]
    metrics_by_type = calculate_change_type_metrics(predicted_diffs, gt_diffs)

    performance = {
        "total_time_seconds": total_time,
        "pages_processed": len(result.pages),
        "time_per_page": total_time / max(1, len(result.pages)),
    }

    vr = VariationResult(
        pair_id=pair_id,
        metrics=metrics,
        metrics_by_type=metrics_by_type,
        performance=performance,
        ground_truth_count=len(ground_truth_changes),
        predicted_count=len(predicted_diffs),
    )

    mismatch_item = build_mismatch_report_item(
        pair_id=pair_id,
        predicted_diffs=predicted_diffs,
        ground_truth_changes=ground_truth_changes,
        tolerance=0.6,
    )
    return vr, mismatch_item


def build_mismatch_report_item(
    *,
    pair_id: str,
    predicted_diffs: List[Diff],
    ground_truth_changes: List[GroundTruthChange],
    tolerance: float = 0.6,
) -> dict:
    """Create a per-variation mismatch report with matches/FP/FN."""
    unmatched_gt = list(ground_truth_changes)
    matches = []
    false_positives = []

    def _match_priority(diff: Diff) -> tuple:
        has_text = bool((diff.old_text or "").strip() or (diff.new_text or "").strip())
        layout_only = (diff.change_type == "layout") and (not has_text)
        return (
            0 if layout_only else 1,
            0 if diff.change_type == "layout" else 1,
            0 if not has_text else 1,
        )

    for pred in sorted(predicted_diffs, key=_match_priority):
        gt, score = match_diff_to_ground_truth_with_score(pred, unmatched_gt, tolerance=tolerance)
        if gt is None:
            false_positives.append(_diff_to_report_dict(pred))
            continue
        matches.append({"score": float(score), "pred": _diff_to_report_dict(pred), "gt": _gt_to_report_dict(gt)})
        unmatched_gt.remove(gt)

    false_negatives = [_gt_to_report_dict(gt) for gt in unmatched_gt]
    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)

    return {
        "pair_id": pair_id,
        "gt_count": len(ground_truth_changes),
        "pred_count": len(predicted_diffs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def get_all_variations() -> List[Path]:
    """Get all variation directories in the synthetic dataset."""
    if not SYNTHETIC_DATASET_DIR.exists():
        return []
    
    variations = []
    for item in sorted(SYNTHETIC_DATASET_DIR.iterdir()):
        if item.is_dir() and item.name.startswith("variation_"):
            variations.append(item)
    
    return variations


@pytest.fixture
def synthetic_variations():
    """Pytest fixture providing all synthetic variations."""
    return get_all_variations()


class TestGoldenEvaluation:
    """Test suite for golden evaluation with synthetic dataset."""
    
    @pytest.mark.skipif(
        not SYNTHETIC_DATASET_DIR.exists(),
        reason="Synthetic dataset not found"
    )
    def test_dataset_exists(self):
        """Test that synthetic dataset exists."""
        assert SYNTHETIC_DATASET_DIR.exists()
        variations = get_all_variations()
        assert len(variations) > 0, "No variations found in synthetic dataset"
    
    @pytest.mark.skipif(
        not SYNTHETIC_DATASET_DIR.exists(),
        reason="Synthetic dataset not found"
    )
    def test_single_variation_evaluation(self):
        """Test evaluation of a single variation."""
        variations = get_all_variations()
        if not variations:
            pytest.skip("No variations available")
        
        # Test first variation
        result = evaluate_variation(variations[0])
        assert result is not None
        assert result.ground_truth_count > 0
        assert result.predicted_count >= 0
        
        # Basic sanity checks
        assert 0.0 <= result.metrics.precision <= 1.0
        assert 0.0 <= result.metrics.recall <= 1.0
        assert 0.0 <= result.metrics.f1_score <= 1.0
    
    @pytest.mark.skipif(
        not SYNTHETIC_DATASET_DIR.exists(),
        reason="Synthetic dataset not found"
    )
    @pytest.mark.slow
    def test_all_variations_evaluation(self):
        """
        Test evaluation of all variations and export results.
        
        This is a comprehensive test that runs the full evaluation suite.
        """
        variations = get_all_variations()
        if not variations:
            pytest.skip("No variations available")
        
        results = []
        mismatch_items = []
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        
        for variation_dir in variations:
            evaluated = evaluate_variation_with_debug(variation_dir)
            if evaluated:
                result, mismatch_item = evaluated
                results.append(result)
                total_precision += result.metrics.precision
                total_recall += result.metrics.recall
                total_f1 += result.metrics.f1_score
                mismatch_items.append(mismatch_item)
        
        assert len(results) > 0, "No variations could be evaluated"
        
        # Calculate averages
        n = len(results)
        avg_precision = total_precision / n
        avg_recall = total_recall / n
        avg_f1 = total_f1 / n
        
        # Export results
        export_data = {
            "summary": {
                "total_variations": n,
                "average_precision": avg_precision,
                "average_recall": avg_recall,
                "average_f1": avg_f1,
            },
            "variations": [r.to_dict() for r in results],
        }
        
        GOLDEN_RESULTS_PATH.write_text(
            json.dumps(export_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Export mismatch report for diagnostics
        MISMATCH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        MISMATCH_REPORT_PATH.write_text(
            json.dumps(
                {
                    "dataset_dir": str(SYNTHETIC_DATASET_DIR),
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "variations": mismatch_items,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        
        # Log results
        print(f"\n=== Golden Evaluation Results ===")
        print(f"Evaluated {n} variations")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Average F1 Score: {avg_f1:.3f}")
        print(f"Results exported to: {GOLDEN_RESULTS_PATH}")
        
        # Assertions for minimum thresholds
        # Note: These thresholds are intentionally lenient for initial testing
        # and can be tightened as the system improves
        assert avg_recall >= 0.3, f"Average recall {avg_recall:.3f} below minimum 0.3"
    
    @pytest.mark.skipif(
        not SYNTHETIC_DATASET_DIR.exists(),
        reason="Synthetic dataset not found"
    )
    def test_ground_truth_loading(self):
        """Test that ground truth can be loaded correctly."""
        variations = get_all_variations()
        if not variations:
            pytest.skip("No variations available")
        
        change_log_path = variations[0] / f"{variations[0].name}_change_log.json"
        if not change_log_path.exists():
            pytest.skip("Change log not found")
        
        changes = load_ground_truth(change_log_path)
        
        assert len(changes) > 0, "No changes loaded from ground truth"
        
        for change in changes:
            assert change.page >= 1
            assert change.diff_type in ("added", "deleted", "modified")
            assert change.change_type in ("content", "formatting", "layout", "visual")
            assert change.description


def run_golden_evaluation() -> Dict:
    """
    Run golden evaluation programmatically and return results.
    
    This function can be called from scripts for batch evaluation.
    
    Returns:
        Dictionary with evaluation results
    """
    variations = get_all_variations()
    
    if not variations:
        return {"error": "No synthetic variations found", "variations": []}
    
    results = []
    for variation_dir in variations:
        result = evaluate_variation(variation_dir)
        if result:
            results.append(result)
    
    if not results:
        return {"error": "No variations could be evaluated", "variations": []}
    
    # Calculate summary statistics
    n = len(results)
    summary = {
        "total_variations": n,
        "average_precision": sum(r.metrics.precision for r in results) / n,
        "average_recall": sum(r.metrics.recall for r in results) / n,
        "average_f1": sum(r.metrics.f1_score for r in results) / n,
        "total_ground_truth_changes": sum(r.ground_truth_count for r in results),
        "total_predicted_diffs": sum(r.predicted_count for r in results),
    }
    
    return {
        "summary": summary,
        "variations": [r.to_dict() for r in results],
    }


if __name__ == "__main__":
    # Run evaluation when executed directly
    import sys
    
    print("=== Golden Evaluation Runner ===")
    print(f"Dataset directory: {SYNTHETIC_DATASET_DIR}")
    
    if not SYNTHETIC_DATASET_DIR.exists():
        print("ERROR: Synthetic dataset not found!")
        print("Run 'python generate_synthetic_dataset.py' first.")
        sys.exit(1)
    
    results = run_golden_evaluation()
    
    if "error" in results:
        print(f"ERROR: {results['error']}")
        sys.exit(1)
    
    summary = results["summary"]
    print(f"\nEvaluated {summary['total_variations']} variations")
    print(f"Ground truth changes: {summary['total_ground_truth_changes']}")
    print(f"Predicted diffs: {summary['total_predicted_diffs']}")
    print(f"\nAverage Precision: {summary['average_precision']:.3f}")
    print(f"Average Recall: {summary['average_recall']:.3f}")
    print(f"Average F1 Score: {summary['average_f1']:.3f}")
    
    # Export results
    GOLDEN_RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults exported to: {GOLDEN_RESULTS_PATH}")

