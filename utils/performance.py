"""Performance profiling utilities."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Generator, List

from config.settings import settings
from utils.logging import logger


@dataclass
class Timing:
    name: str
    duration: float
    metadata: dict = field(default_factory=dict)


_timings: List[Timing] = []


@contextmanager
def track_time(name: str, **metadata) -> Generator[Timing, None, None]:
    """Context manager to track execution time."""
    timing = Timing(name=name, duration=0, metadata=metadata)
    start = time.perf_counter()
    try:
        yield timing
    finally:
        timing.duration = time.perf_counter() - start
        _timings.append(timing)
        logger.debug("Timing: %s took %.3f seconds", name, timing.duration)


def time_it(fn: Callable, *args, **kwargs):
    """Time a function call."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    duration = time.perf_counter() - start
    return result, duration


def get_timings() -> List[Timing]:
    """Get all recorded timings."""
    return _timings.copy()


def clear_timings() -> None:
    """Clear recorded timings."""
    _timings.clear()


def check_performance_target(page_count: int, total_time: float) -> bool:
    """Check if performance meets the target (<3s per page)."""
    if page_count == 0:
        return True
    
    time_per_page = total_time / page_count
    target = settings.seconds_per_page_target
    
    meets_target = time_per_page < target
    if not meets_target:
        logger.warning(
            "Performance target not met: %.2fs per page (target: %.2fs)",
            time_per_page,
            target,
        )
    else:
        logger.info(
            "Performance target met: %.2fs per page (target: %.2fs)",
            time_per_page,
            target,
        )
    
    return meets_target


def print_performance_summary() -> None:
    """Print a summary of all recorded timings."""
    if not _timings:
        logger.info("No timings recorded")
        return
    
    logger.info("Performance Summary:")
    total = sum(t.duration for t in _timings)
    for timing in _timings:
        percentage = (timing.duration / total * 100) if total > 0 else 0
        logger.info(
            "  %s: %.3fs (%.1f%%)",
            timing.name,
            timing.duration,
            percentage,
        )
    logger.info("  Total: %.3fs", total)
