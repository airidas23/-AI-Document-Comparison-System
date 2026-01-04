#!/usr/bin/env python3
"""
Evaluation runner: Compare synthetic PDF pairs and compute F1/latency metrics.

Reads ground truth from synthetic dataset, runs comparison pipeline,
and outputs evaluation report with precision/recall/F1 scores.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Add repository root to path so imports work when running as a script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from comparison.models import Diff
from pipeline import compare_pdfs, ComparisonPipeline, PipelineConfig
from utils.logging import configure_logging, logger
from utils.metrics import (
    ChangeDetectionMetrics,
    PerformanceMetrics,
    calculate_change_detection_metrics,
    calculate_performance_metrics,
    calculate_change_type_metrics,
)

configure_logging()


@dataclass
class EvaluationResult:
    """Result of evaluating a single PDF pair."""
    pair_id: str
    original_pdf: str
    modified_pdf: str
    ground_truth_count: int
    detected_count: int
    metrics: ChangeDetectionMetrics
    metrics_by_type: Dict[str, ChangeDetectionMetrics]
    performance: PerformanceMetrics
    errors: List[str]


def load_ground_truth(change_log_path: Path) -> List[Diff]:
    """Load ground truth diffs from a change log JSON file."""
    if not change_log_path.exists():
        logger.warning("Change log not found: %s", change_log_path)
        return []
    
    try:
        with open(change_log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        diffs = []
        changes = data.get("changes", [])
        
        for change in changes:
            diff = Diff(
                page_num=change.get("page", 1),
                diff_type=change.get("diff_type", "modified"),
                change_type=change.get("change_type", "content"),
                # Synthetic change logs use before/after fields.
                old_text=change.get("before", change.get("old_text")),
                new_text=change.get("after", change.get("new_text")),
                bbox=change.get("bbox"),
                confidence=1.0,  # Ground truth is certain
                metadata={
                    **(change.get("metadata") or {}),
                    "region": change.get("region"),
                    "description": change.get("description"),
                    "severity": change.get("severity"),
                },
            )
            diffs.append(diff)
        
        return diffs
    
    except Exception as e:
        logger.error("Failed to load change log %s: %s", change_log_path, e)
        return []


def evaluate_pair(
    original_pdf: Path,
    modified_pdf: Path,
    ground_truth: List[Diff],
    pair_id: str,
    config: Optional[PipelineConfig] = None,
) -> EvaluationResult:
    """
    Evaluate comparison pipeline on a single PDF pair.
    
    Args:
        original_pdf: Path to original PDF
        modified_pdf: Path to modified PDF
        ground_truth: List of ground truth diffs
        pair_id: Identifier for this pair
        config: Pipeline configuration
        
    Returns:
        EvaluationResult with metrics
    """
    errors = []
    detected_diffs = []
    total_time = 0.0
    pages_processed = 0
    
    try:
        # Run comparison pipeline
        start_time = time.time()
        
        pipeline = ComparisonPipeline(config or PipelineConfig())
        result = pipeline.compare(original_pdf, modified_pdf)
        
        total_time = time.time() - start_time
        detected_diffs = result.diffs
        pages_processed = len(result.pages)
        
    except Exception as e:
        logger.error("Comparison failed for %s: %s", pair_id, e)
        errors.append(str(e))
    
    # Calculate metrics
    metrics = calculate_change_detection_metrics(detected_diffs, ground_truth)
    metrics_by_type = calculate_change_type_metrics(detected_diffs, ground_truth)
    performance = calculate_performance_metrics(
        total_time, 
        pages_processed,
        target_seconds_per_page=3.0,  # 3s/page target
    )
    
    return EvaluationResult(
        pair_id=pair_id,
        original_pdf=str(original_pdf),
        modified_pdf=str(modified_pdf),
        ground_truth_count=len(ground_truth),
        detected_count=len(detected_diffs),
        metrics=metrics,
        metrics_by_type=metrics_by_type,
        performance=performance,
        errors=errors,
    )


def evaluate_dataset(
    dataset_dir: Path,
    config: Optional[PipelineConfig] = None,
    *,
    scanned: bool = False,
) -> List[EvaluationResult]:
    """
    Evaluate comparison pipeline on entire synthetic dataset.
    
    Args:
        dataset_dir: Directory containing synthetic PDF pairs
        config: Pipeline configuration
        
    Returns:
        List of EvaluationResult objects
    """
    results = []
    
    # Find all PDF pairs by looking for change logs.
    # Synthetic dataset is typically organized as nested folders like `variation_01/`.
    change_logs = sorted(dataset_dir.rglob("*_change_log.json"))
    
    if not change_logs:
        logger.warning("No change logs found in %s", dataset_dir)
        return results
    
    logger.info("Found %d evaluation pairs", len(change_logs))
    
    for i, change_log_path in enumerate(change_logs, 1):
        # Parse pair ID from change log filename
        pair_id = change_log_path.stem.replace("_change_log", "")

        pair_dir = change_log_path.parent
        
        # Load ground truth
        ground_truth = load_ground_truth(change_log_path)

        # Load change log payload for path resolution (especially for scanned PDFs)
        try:
            with open(change_log_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {}
        
        def _pick_payload_path(key: str) -> Optional[Path]:
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return Path(val)
            return None

        if scanned:
            # Prefer explicit scanned paths from the change log.
            original_pdf = _pick_payload_path("original_scanned_pdf") or (pair_dir / f"{pair_id}_original_scanned.pdf")
            modified_pdf = _pick_payload_path("modified_scanned_pdf") or (pair_dir / f"{pair_id}_modified_scanned.pdf")
        else:
            # Convention: pair_X_original.pdf, pair_X_modified.pdf
            original_pdf = _pick_payload_path("original_pdf") or (pair_dir / f"{pair_id}_original.pdf")
            modified_pdf = _pick_payload_path("modified_pdf") or (pair_dir / f"{pair_id}_modified.pdf")
        
        # Alternative naming: base_document.pdf for original
        if not original_pdf.exists():
            original_pdf = dataset_dir / "base_document.pdf"
        
        if not original_pdf.exists() or not modified_pdf.exists():
            logger.warning("PDFs not found for %s, skipping", pair_id)
            continue
        
        logger.info("[%d/%d] Evaluating %s...", i, len(change_logs), pair_id)
        
        result = evaluate_pair(
            original_pdf=original_pdf,
            modified_pdf=modified_pdf,
            ground_truth=ground_truth,
            pair_id=pair_id,
            config=config,
        )
        
        results.append(result)
        
        # Log progress
        logger.info(
            "  → F1: %.3f | Precision: %.3f | Recall: %.3f | Time: %.2fs",
            result.metrics.f1_score,
            result.metrics.precision,
            result.metrics.recall,
            result.performance.total_time,
        )
    
    return results


def aggregate_metrics(results: List[EvaluationResult]) -> Dict:
    """Aggregate metrics across all evaluation results."""
    if not results:
        return {}
    
    # Aggregate overall metrics
    total_ground_truth = sum(r.ground_truth_count for r in results)
    total_detected = sum(r.detected_count for r in results)
    
    precisions = [r.metrics.precision for r in results]
    recalls = [r.metrics.recall for r in results]
    f1_scores = [r.metrics.f1_score for r in results]
    
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    # Aggregate performance
    total_time = sum(r.performance.total_time for r in results)
    total_pages = sum(r.performance.pages_processed for r in results)
    avg_time_per_page = total_time / max(1, total_pages)
    meets_target = avg_time_per_page < 3.0
    
    # Aggregate by change type
    metrics_by_type = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})
    for result in results:
        for change_type, m in result.metrics_by_type.items():
            metrics_by_type[change_type]["precision"].append(m.precision)
            metrics_by_type[change_type]["recall"].append(m.recall)
            metrics_by_type[change_type]["f1"].append(m.f1_score)
    
    aggregated_by_type = {}
    for change_type, values in metrics_by_type.items():
        aggregated_by_type[change_type] = {
            "avg_precision": sum(values["precision"]) / len(values["precision"]),
            "avg_recall": sum(values["recall"]) / len(values["recall"]),
            "avg_f1": sum(values["f1"]) / len(values["f1"]),
            "sample_count": len(values["f1"]),
        }
    
    # Error summary
    pairs_with_errors = sum(1 for r in results if r.errors)
    all_errors = [e for r in results for e in r.errors]
    
    return {
        "summary": {
            "pairs_evaluated": len(results),
            "total_ground_truth_diffs": total_ground_truth,
            "total_detected_diffs": total_detected,
            "pairs_with_errors": pairs_with_errors,
        },
        "overall_metrics": {
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_f1_score": round(avg_f1, 4),
            "min_f1": round(min(f1_scores), 4),
            "max_f1": round(max(f1_scores), 4),
        },
        "performance": {
            "total_time_seconds": round(total_time, 2),
            "total_pages_processed": total_pages,
            "avg_time_per_page": round(avg_time_per_page, 2),
            "meets_3s_target": meets_target,
            "pages_per_minute": round((total_pages / max(0.001, total_time)) * 60, 1),
        },
        "metrics_by_change_type": aggregated_by_type,
        "errors": all_errors[:10] if all_errors else [],  # First 10 errors
    }


def generate_report(
    results: List[EvaluationResult],
    aggregated: Dict,
    output_path: Path,
) -> None:
    """Generate markdown evaluation report."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Report: Document Comparison Pipeline\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        summary = aggregated.get("summary", {})
        f.write(f"- **Pairs Evaluated:** {summary.get('pairs_evaluated', 0)}\n")
        f.write(f"- **Total Ground Truth Diffs:** {summary.get('total_ground_truth_diffs', 0)}\n")
        f.write(f"- **Total Detected Diffs:** {summary.get('total_detected_diffs', 0)}\n")
        f.write(f"- **Pairs with Errors:** {summary.get('pairs_with_errors', 0)}\n\n")
        
        # Overall Metrics
        f.write("## Overall Metrics\n\n")
        metrics = aggregated.get("overall_metrics", {})
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Average F1 Score** | {metrics.get('avg_f1_score', 0):.4f} |\n")
        f.write(f"| **Average Precision** | {metrics.get('avg_precision', 0):.4f} |\n")
        f.write(f"| **Average Recall** | {metrics.get('avg_recall', 0):.4f} |\n")
        f.write(f"| Min F1 | {metrics.get('min_f1', 0):.4f} |\n")
        f.write(f"| Max F1 | {metrics.get('max_f1', 0):.4f} |\n\n")
        
        # Performance
        f.write("## Performance\n\n")
        perf = aggregated.get("performance", {})
        target_met = "✅ Yes" if perf.get("meets_3s_target") else "❌ No"
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Avg Time per Page** | {perf.get('avg_time_per_page', 0):.2f}s |\n")
        f.write(f"| **Meets 3s/page Target** | {target_met} |\n")
        f.write(f"| Total Time | {perf.get('total_time_seconds', 0):.2f}s |\n")
        f.write(f"| Total Pages | {perf.get('total_pages_processed', 0)} |\n")
        f.write(f"| Pages per Minute | {perf.get('pages_per_minute', 0):.1f} |\n\n")
        
        # Metrics by Change Type
        f.write("## Metrics by Change Type\n\n")
        by_type = aggregated.get("metrics_by_change_type", {})
        if by_type:
            f.write("| Change Type | Avg F1 | Avg Precision | Avg Recall | Samples |\n")
            f.write("|-------------|--------|---------------|------------|--------|\n")
            for change_type, m in sorted(by_type.items()):
                f.write(f"| {change_type} | {m['avg_f1']:.4f} | {m['avg_precision']:.4f} | {m['avg_recall']:.4f} | {m['sample_count']} |\n")
            f.write("\n")
        
        # Per-Pair Results
        f.write("## Per-Pair Results\n\n")
        f.write("| Pair | GT | Detected | F1 | Precision | Recall | Time |\n")
        f.write("|------|----|---------|----|-----------|--------|------|\n")
        for r in results:
            status = "❌" if r.errors else ""
            f.write(
                f"| {r.pair_id} {status} | {r.ground_truth_count} | {r.detected_count} | "
                f"{r.metrics.f1_score:.3f} | {r.metrics.precision:.3f} | {r.metrics.recall:.3f} | "
                f"{r.performance.total_time:.2f}s |\n"
            )
        f.write("\n")
        
        # Errors
        errors = aggregated.get("errors", [])
        if errors:
            f.write("## Errors\n\n")
            for err in errors:
                f.write(f"- {err}\n")
            f.write("\n")
        
        # Targets
        f.write("## Target Thresholds\n\n")
        f.write("From `docs/METRICS_AND_THRESHOLDS.md`:\n\n")
        f.write("- **F1 Score Target:** ≥0.85\n")
        f.write("- **Precision Target:** ≥0.90\n")
        f.write("- **Recall Target:** ≥0.80\n")
        f.write("- **Performance Target:** <3s per page\n")
    
    logger.info("Report written to: %s", output_path)


def main():
    """Main evaluation runner."""
    import argparse

    default_dataset_dir = project_root / "data" / "synthetic" / "dataset"

    parser = argparse.ArgumentParser(
        prog="run_evaluation.py",
        description=(
            "Evaluate the comparison pipeline on the synthetic dataset and generate "
            "F1/Precision/Recall/Latency metrics + a Markdown report."
        ),
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(default_dataset_dir),
        help="Path to synthetic dataset directory (default: data/synthetic/dataset)",
    )
    parser.add_argument(
        "--ocr-mode",
        choices=["auto", "hybrid", "ocr_only"],
        default="auto",
        help="OCR enhancement mode used by the pipeline",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["paddle", "tesseract", "deepseek"],
        default=None,
        help=(
            "OCR engine override used by the pipeline. "
            "If omitted, uses config/settings defaults. "
            "DeepSeek is supported but may be slow/unavailable depending on hardware."
        ),
    )
    parser.add_argument(
        "--no-layout",
        action="store_true",
        help="Disable layout analysis during evaluation (faster, avoids layout model deps)",
    )
    parser.add_argument(
        "--scanned",
        action="store_true",
        help="Evaluate scanned PDFs (uses *_scanned.pdf paths from each change log)",
    )

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    
    if not dataset_dir.exists():
        logger.error("Dataset directory not found: %s", dataset_dir)
        logger.info("Generate synthetic dataset first: python generate_synthetic_dataset.py")
        return 1
    
    logger.info("=" * 60)
    logger.info("EVALUATION RUNNER")
    logger.info("=" * 60)
    logger.info("Dataset: %s", dataset_dir)

    # If the user explicitly requests an OCR engine, make the run deterministic.
    # Otherwise, scanned OCR routing may auto-fallback to a different engine.
    if args.ocr_engine:
        from config.settings import settings

        settings.ocr_engine = args.ocr_engine
        settings.ocr_fallback_enabled = False

        # For scanned PDFs, also disable the multi-engine router.
        if args.scanned:
            settings.ocr_scanned_policy = "strict"
            settings.ocr_engine_priority = [args.ocr_engine]
            settings.ocr_scanned_fallback_chain = [args.ocr_engine]
            settings.ocr_scanned_default_chain = [args.ocr_engine]
            if args.ocr_engine != "deepseek":
                settings.deepseek_enabled = False
    
    # Configure pipeline
    config = PipelineConfig(
        ocr_mode=args.ocr_mode,
        ocr_engine=args.ocr_engine,
        run_layout_analysis=not args.no_layout,
    )
    
    # Run evaluation
    results = evaluate_dataset(dataset_dir, config, scanned=args.scanned)
    
    if not results:
        logger.error("No evaluation results!")
        return 1
    
    # Aggregate metrics
    aggregated = aggregate_metrics(results)
    
    # Save JSON results
    # Keep backwards-compatible filenames by default, but avoid clobbering
    # when users run per-engine or scanned evaluations.
    suffix_parts = []
    if args.scanned:
        suffix_parts.append("scanned")
    if args.ocr_engine:
        suffix_parts.append(args.ocr_engine)
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    json_path = dataset_dir / f"evaluation_results{suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # Convert EvaluationResult to serializable format
        results_data = []
        for r in results:
            results_data.append({
                "pair_id": r.pair_id,
                "original_pdf": r.original_pdf,
                "modified_pdf": r.modified_pdf,
                "ground_truth_count": r.ground_truth_count,
                "detected_count": r.detected_count,
                "metrics": r.metrics.to_dict(),
                "performance": r.performance.to_dict(),
                "errors": r.errors,
            })
        
        json.dump({
            "results": results_data,
            "aggregated": aggregated,
        }, f, indent=2, default=str)
    
    logger.info("JSON results saved to: %s", json_path)
    
    # Generate markdown report
    report_path = dataset_dir / f"evaluation_report{suffix}.md"
    generate_report(results, aggregated, report_path)
    
    # Also update TEST_RESULTS.md if it exists
    test_results_path = project_root / "tests" / "TEST_RESULTS.md"
    if test_results_path.parent.exists():
        # Append evaluation section
        with open(test_results_path, "a", encoding="utf-8") as f:
            f.write("\n\n---\n\n")
            f.write(f"## Synthetic Dataset Evaluation ({time.strftime('%Y-%m-%d')})\n\n")
            metrics = aggregated.get("overall_metrics", {})
            perf = aggregated.get("performance", {})
            f.write(f"- **F1 Score:** {metrics.get('avg_f1_score', 0):.4f}\n")
            f.write(f"- **Precision:** {metrics.get('avg_precision', 0):.4f}\n")
            f.write(f"- **Recall:** {metrics.get('avg_recall', 0):.4f}\n")
            f.write(f"- **Avg Time/Page:** {perf.get('avg_time_per_page', 0):.2f}s\n")
            f.write(f"\nFull report: [evaluation_report.md](../data/synthetic/dataset/evaluation_report.md)\n")
        logger.info("Updated: %s", test_results_path)
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    metrics = aggregated.get("overall_metrics", {})
    perf = aggregated.get("performance", {})
    logger.info("Pairs evaluated: %d", len(results))
    logger.info("")
    logger.info("OVERALL METRICS:")
    logger.info("  F1 Score:  %.4f (target: ≥0.85)", metrics.get("avg_f1_score", 0))
    logger.info("  Precision: %.4f (target: ≥0.90)", metrics.get("avg_precision", 0))
    logger.info("  Recall:    %.4f (target: ≥0.80)", metrics.get("avg_recall", 0))
    logger.info("")
    logger.info("PERFORMANCE:")
    logger.info("  Avg time/page: %.2fs (target: <3s)", perf.get("avg_time_per_page", 0))
    target_met = "✅" if perf.get("meets_3s_target") else "❌"
    logger.info("  Target met: %s", target_met)
    logger.info("")
    logger.info("Reports:")
    logger.info("  - %s", json_path)
    logger.info("  - %s", report_path)
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
