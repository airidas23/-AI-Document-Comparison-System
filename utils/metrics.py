"""Evaluation metrics and quality assessment utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from comparison.models import Diff


def _norm_text(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _text_similarity(a: str, b: str) -> float:
    """Return similarity in [0, 1]. Uses RapidFuzz if available."""
    a = _norm_text(a)
    b = _norm_text(b)
    if not a or not b:
        return 0.0

    try:
        from rapidfuzz import fuzz

        # token_set_ratio is robust to small reorderings / punctuation.
        return float(fuzz.token_set_ratio(a, b)) / 100.0
    except Exception:
        # Fallback: simple substring overlap.
        if a in b or b in a:
            return 1.0
        return 0.0


@dataclass
class ChangeDetectionMetrics:
    """Metrics for change detection evaluation."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
        }


@dataclass
class AlignmentMetrics:
    """Metrics for page/section alignment accuracy."""
    alignment_accuracy: float
    correct_alignments: int
    total_alignments: int
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "alignment_accuracy": self.alignment_accuracy,
            "correct_alignments": self.correct_alignments,
            "total_alignments": self.total_alignments,
            "error_rate": 1.0 - self.alignment_accuracy,
        }


@dataclass
class PerformanceMetrics:
    """Performance timing metrics."""
    time_per_page: float
    total_time: float
    pages_processed: int
    meets_target: bool
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "time_per_page": self.time_per_page,
            "total_time": self.total_time,
            "pages_processed": self.pages_processed,
            "meets_target": self.meets_target,
            "pages_per_minute": (self.pages_processed / self.total_time * 60) if self.total_time > 0 else 0.0,
        }


def calculate_change_detection_metrics(
    predicted_diffs: List[Diff],
    ground_truth_diffs: List[Diff],
    tolerance: float = 0.1,
) -> ChangeDetectionMetrics:
    """
    Calculate precision, recall, F1 score, and accuracy for change detection.
    
    Args:
        predicted_diffs: List of detected diffs
        ground_truth_diffs: List of ground truth diffs
        tolerance: Position tolerance for matching (normalized coordinates)
    
    Returns:
        ChangeDetectionMetrics object
    """
    if not ground_truth_diffs:
        if not predicted_diffs:
            return ChangeDetectionMetrics(precision=1.0, recall=1.0, f1_score=1.0, accuracy=1.0)
        return ChangeDetectionMetrics(precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0)
    
    if not predicted_diffs:
        return ChangeDetectionMetrics(precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0)
    
    # Match predicted diffs to ground truth
    matched_predicted = set()
    matched_ground_truth = set()
    
    for pred_idx, pred_diff in enumerate(predicted_diffs):
        for gt_idx, gt_diff in enumerate(ground_truth_diffs):
            if gt_idx in matched_ground_truth:
                continue
            if _match_diff(pred_diff, gt_diff, tolerance):
                matched_predicted.add(pred_idx)
                matched_ground_truth.add(gt_idx)
                break
    
    tp = len(matched_predicted)  # True positives
    fp = len(predicted_diffs) - tp  # False positives
    fn = len(ground_truth_diffs) - len(matched_ground_truth)  # False negatives
    
    # Calculate metrics
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


def _match_diff(pred_diff: Diff, gt_diff: Diff, tolerance: float) -> bool:
    """Check if two diffs match (same page, similar position, same type)."""
    # Same page
    if pred_diff.page_num != gt_diff.page_num:
        return False
    
    # Same change type
    if pred_diff.change_type != gt_diff.change_type:
        return False
    
    # Diff type compatibility
    if pred_diff.diff_type != gt_diff.diff_type:
        # Allow flexibility: many detectors collapse changes into "modified".
        if not (pred_diff.diff_type == "modified" or gt_diff.diff_type == "modified"):
            return False
    
    # Similar position (if both have bboxes)
    if pred_diff.bbox and gt_diff.bbox:
        pred_x = pred_diff.bbox.get("x", 0.0)
        pred_y = pred_diff.bbox.get("y", 0.0)
        gt_x = gt_diff.bbox.get("x", 0.0)
        gt_y = gt_diff.bbox.get("y", 0.0)
        
        distance = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
        if distance > tolerance:
            return False
    else:
        # If bbox is missing on either side (common for synthetic ground truth),
        # fall back to text-based matching for content/formatting diffs.
        if pred_diff.change_type in ("content", "formatting"):
            pred_old = _norm_text(pred_diff.old_text)
            pred_new = _norm_text(pred_diff.new_text)
            gt_old = _norm_text(gt_diff.old_text)
            gt_new = _norm_text(gt_diff.new_text)

            # If ground truth has no text, accept page+type match.
            if not (gt_old or gt_new):
                return True

            pairs = [
                (pred_old, gt_old),
                (pred_new, gt_new),
                (pred_old, gt_new),
                (pred_new, gt_old),
            ]
            best_sim = 0.0
            for a, b in pairs:
                if not a or not b:
                    continue
                # Substring shortcut for common partial matches
                if a[:30] and (a[:30] in b or b[:30] in a):
                    return True
                best_sim = max(best_sim, _text_similarity(a, b))

            # Relaxed threshold: synthetic GT may differ in punctuation/casing.
            return best_sim >= 0.6
    
    return True


def calculate_alignment_accuracy(
    alignment_map: Dict[int, tuple[int, float]],
    ground_truth_map: Dict[int, int],
) -> AlignmentMetrics:
    """
    Calculate alignment accuracy.
    
    Args:
        alignment_map: Predicted alignment map (page_a -> (page_b, confidence))
        ground_truth_map: Ground truth alignment map (page_a -> page_b)
    
    Returns:
        AlignmentMetrics object
    """
    if not ground_truth_map:
        return AlignmentMetrics(alignment_accuracy=0.0, correct_alignments=0, total_alignments=0)
    
    correct = 0
    total = len(ground_truth_map)
    
    for page_a, expected_page_b in ground_truth_map.items():
        if page_a in alignment_map:
            predicted_page_b, _ = alignment_map[page_a]
            if predicted_page_b == expected_page_b:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return AlignmentMetrics(
        alignment_accuracy=accuracy,
        correct_alignments=correct,
        total_alignments=total,
    )


def calculate_performance_metrics(
    total_time: float,
    pages_processed: int,
    target_seconds_per_page: float = 3.0,
) -> PerformanceMetrics:
    """
    Calculate performance metrics.
    
    Args:
        total_time: Total processing time in seconds
        pages_processed: Number of pages processed
        target_seconds_per_page: Target time per page
    
    Returns:
        PerformanceMetrics object
    """
    time_per_page = total_time / pages_processed if pages_processed > 0 else 0.0
    meets_target = time_per_page < target_seconds_per_page
    
    return PerformanceMetrics(
        time_per_page=time_per_page,
        total_time=total_time,
        pages_processed=pages_processed,
        meets_target=meets_target,
    )


def calculate_change_type_metrics(
    predicted_diffs: List[Diff],
    ground_truth_diffs: List[Diff],
) -> Dict[str, ChangeDetectionMetrics]:
    """
    Calculate metrics broken down by change type.
    
    Returns:
        Dictionary mapping change_type to metrics
    """
    change_types = set(d.change_type for d in predicted_diffs + ground_truth_diffs)
    metrics_by_type = {}
    
    for change_type in change_types:
        pred_filtered = [d for d in predicted_diffs if d.change_type == change_type]
        gt_filtered = [d for d in ground_truth_diffs if d.change_type == change_type]
        
        metrics = calculate_change_detection_metrics(pred_filtered, gt_filtered)
        metrics_by_type[change_type] = metrics
    
    return metrics_by_type

