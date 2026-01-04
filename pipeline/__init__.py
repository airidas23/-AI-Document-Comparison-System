"""Pipeline module - orchestrates end-to-end document comparison."""
from pipeline.compare_pdfs import (
    compare_pdfs,
    extract_single_pdf,
    ComparisonPipeline,
    PipelineConfig,
    PipelineMetrics,
)

__all__ = [
    "compare_pdfs",
    "extract_single_pdf",
    "ComparisonPipeline",
    "PipelineConfig",
    "PipelineMetrics",
]
