"""Generate figures used by docs/TESTING_PLAN.md.

This script is intentionally conservative:
- It only uses metrics already present in JSON artifacts.
- If an expected input artifact is missing, the corresponding figure is skipped.

Run:
  .venv/bin/python scripts/generate_testing_plan_assets.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class EvalGroup:
    label: str
    precision: float
    recall: float
    f1: float
    avg_time_per_page_s: float


ChangeTypeF1 = dict[str, float]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"[skip] Missing input: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> float:
    if value is None:
        raise ValueError("Missing numeric value")
    return float(value)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Empty list")
    # Simple, dependency-free percentile (linear interpolation).
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return values_sorted[0]
    rank = (len(values_sorted) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(values_sorted) - 1)
    w = rank - lo
    return values_sorted[lo] * (1.0 - w) + values_sorted[hi] * w


def _load_golden_group(path: Path) -> EvalGroup:
    data = _read_json(path)
    if data is None:
        raise FileNotFoundError(path)

    summary = data.get("summary") or {}
    precision = _safe_float(summary.get("average_precision"))
    recall = _safe_float(summary.get("average_recall"))
    f1 = _safe_float(summary.get("average_f1"))

    times: list[float] = []
    for variation in data.get("variations", []):
        perf = (variation or {}).get("performance") or {}
        if "time_per_page" in perf:
            times.append(_safe_float(perf["time_per_page"]))

    if not times:
        raise ValueError("Golden results missing per-variation performance.time_per_page")

    avg_time = sum(times) / len(times)

    # p95 is useful in docs, but plots use average only.
    p95 = _percentile(times, 95)
    print(f"[golden] mean={avg_time:.3f}s/page, p95={p95:.3f}s/page ({len(times)} samples)")

    return EvalGroup(
        label="Golden (digital, PyMuPDF)",
        precision=precision,
        recall=recall,
        f1=f1,
        avg_time_per_page_s=avg_time,
    )


def _load_golden_f1_by_change_type(path: Path) -> ChangeTypeF1:
    data = _read_json(path)
    if data is None:
        raise FileNotFoundError(path)

    buckets: dict[str, list[float]] = {}
    for variation in data.get("variations", []):
        by_type = (variation or {}).get("metrics_by_type") or {}
        for change_type, metrics in by_type.items():
            if metrics and "f1_score" in metrics:
                buckets.setdefault(change_type, []).append(_safe_float(metrics["f1_score"]))

    if not buckets:
        raise ValueError("Golden results missing metrics_by_type")

    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


def _load_eval_results_group(path: Path, label: str) -> EvalGroup:
    data = _read_json(path)
    if data is None:
        raise FileNotFoundError(path)

    aggregated = data.get("aggregated") or {}
    overall = aggregated.get("overall_metrics") or {}
    perf = aggregated.get("performance") or {}

    return EvalGroup(
        label=label,
        precision=_safe_float(overall.get("avg_precision")),
        recall=_safe_float(overall.get("avg_recall")),
        f1=_safe_float(overall.get("avg_f1_score")),
        avg_time_per_page_s=_safe_float(perf.get("avg_time_per_page")),
    )


def _load_eval_results_f1_by_change_type(path: Path) -> ChangeTypeF1:
    data = _read_json(path)
    if data is None:
        raise FileNotFoundError(path)

    aggregated = data.get("aggregated") or {}
    by_type = aggregated.get("metrics_by_change_type") or {}

    out: ChangeTypeF1 = {}
    for change_type, metrics in by_type.items():
        if metrics and "avg_f1" in metrics:
            out[change_type] = _safe_float(metrics["avg_f1"])

    if not out:
        raise ValueError(f"Missing aggregated.metrics_by_change_type in {path}")

    return out


def _plot_prf1(groups: list[EvalGroup], out_path: Path) -> None:
    labels = [g.label for g in groups]
    x = list(range(len(groups)))

    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar([i - width for i in x], [g.precision for g in groups], width, label="Precision")
    ax.bar(x, [g.recall for g in groups], width, label="Recall")
    ax.bar([i + width for i in x], [g.f1 for g in groups], width, label="F1")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall Precision / Recall / F1 by dataset & engine")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[write] {out_path}")


def _plot_latency(groups: list[EvalGroup], out_path: Path) -> None:
    labels = [g.label for g in groups]
    times = [g.avg_time_per_page_s for g in groups]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(labels)))
    bars = ax.bar(x, times)

    ax.set_ylabel("Avg time per page (s)")
    ax.set_title("End-to-end latency by dataset & engine")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")

    for bar, val in zip(bars, times, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}s", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[write] {out_path}")


def _plot_threshold_overview(groups: list[EvalGroup], out_path: Path) -> None:
    """Plot F1 and latency vs. acceptance thresholds.

    Uses only already-available (aggregated) metrics.
    """

    labels = [g.label for g in groups]
    x = list(range(len(labels)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # F1 overview
    f1_values = [g.f1 for g in groups]
    ax1.bar(x, f1_values)
    ax1.axhline(0.85, color="red", linestyle="--", linewidth=1, label="Target F1 ≥ 0.85")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("F1")
    ax1.set_title("Quality & performance vs targets (from available artifacts)")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8)

    # Latency overview
    latency_values = [g.avg_time_per_page_s for g in groups]
    ax2.bar(x, latency_values)
    ax2.axhline(3.0, color="red", linestyle="--", linewidth=1, label="Target latency < 3.0s/page")
    ax2.set_ylabel("Avg time/page (s)")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[write] {out_path}")


def _plot_ocr_benchmark(benchmark_path: Path, out_path: Path) -> None:
    data = _read_json(benchmark_path)
    if data is None:
        raise FileNotFoundError(benchmark_path)

    # Collect times (seconds) for each engine in each scenario.
    scenarios = [
        ("digital_pdf", "Digital PDF (extraction only)"),
        ("scanned_pdf", "Scanned PDF (OCR only)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, (key, title) in zip(axes, scenarios, strict=True):
        scenario = data.get(key) or {}
        engines = list(scenario.keys())
        times = [_safe_float((scenario.get(e) or {}).get("time")) for e in engines]

        ax.bar(engines, times)
        ax.set_title(title)
        ax.set_ylabel("Time (s)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        # Log scale helps make PyMuPDF visible vs OCR.
        ax.set_yscale("log")
        for i, val in enumerate(times):
            ax.text(i, val, f"{val:.3f}s", ha="center", va="bottom", fontsize=8)

    fig.suptitle("OCR micro-benchmark (log scale)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[write] {out_path}")


def _plot_f1_by_change_type(groups: list[EvalGroup], f1_maps: dict[str, ChangeTypeF1], out_path: Path) -> None:
    # Keep a consistent order of categories used across docs.
    categories = ["content", "formatting", "layout", "visual"]

    present_labels = [g.label for g in groups if g.label in f1_maps]
    if not present_labels:
        print("[skip] No per-change-type maps found; skipping f1_by_change_type plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(categories)))

    # Dynamic width based on number of groups.
    n = len(present_labels)
    width = min(0.18, 0.8 / max(1, n))
    offsets = [(i - (n - 1) / 2) * width for i in range(n)]

    for label, offset in zip(present_labels, offsets, strict=True):
        m = f1_maps[label]
        values = [m.get(cat, float("nan")) for cat in categories]
        ax.bar([xi + offset for xi in x], values, width, label=label)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("F1 by change category (from available artifacts)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[write] {out_path}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    assets_dir = repo_root / "docs" / "assets"

    groups: list[EvalGroup] = []
    f1_by_type: dict[str, ChangeTypeF1] = {}

    # Digital/native evaluations
    try:
        golden_path = repo_root / "tests" / "golden_results.json"
        golden_group = _load_golden_group(golden_path)
        groups.append(golden_group)
        f1_by_type[golden_group.label] = _load_golden_f1_by_change_type(golden_path)
    except FileNotFoundError:
        pass

    try:
        synthetic_path = repo_root / "data" / "synthetic" / "dataset" / "evaluation_results.json"
        synthetic_group = _load_eval_results_group(synthetic_path, "Synthetic (digital, PyMuPDF)")
        groups.append(synthetic_group)
        f1_by_type[synthetic_group.label] = _load_eval_results_f1_by_change_type(synthetic_path)
    except FileNotFoundError:
        pass

    # Scanned evaluations (forced engine)
    for engine in ("tesseract", "paddle"):
        try:
            scanned_path = (
                repo_root
                / "data"
                / "synthetic"
                / "test_scanned_dataset"
                / f"evaluation_results_scanned_{engine}.json"
            )
            scanned_group = _load_eval_results_group(scanned_path, f"Scanned ({engine})")
            groups.append(scanned_group)
            f1_by_type[scanned_group.label] = _load_eval_results_f1_by_change_type(scanned_path)
        except FileNotFoundError:
            pass

    if groups:
        _plot_prf1(groups, assets_dir / "metrics_prf1_overall.png")
        _plot_latency(groups, assets_dir / "latency_end_to_end.png")
        _plot_f1_by_change_type(groups, f1_by_type, assets_dir / "f1_by_change_type.png")
        _plot_threshold_overview(groups, assets_dir / "threshold_overview.png")
    else:
        print("[skip] No evaluation groups found; skipping PRF1/latency plots")

    # OCR micro-benchmark
    try:
        _plot_ocr_benchmark(repo_root / "benchmark" / "benchmark_results.json", assets_dir / "ocr_benchmark_latency.png")
    except FileNotFoundError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
