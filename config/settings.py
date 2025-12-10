"""Configuration management for model paths, thresholds, and performance settings."""
from __future__ import annotations

from functools import lru_cache

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for Pydantic v1
    from pydantic import BaseSettings  # type: ignore

from typing import List, Optional

from pydantic import Field


class Settings(BaseSettings):
    # Model paths
    deepseek_ocr_model_path: str = Field(
        default="models/deepseek-ocr",
        description="Local path to DeepSeek-OCR model weights",
    )
    sentence_transformer_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name or path for sentence-transformer embeddings",
    )
    layoutparser_model: Optional[str] = Field(
        default=None,
        description="LayoutParser model name (e.g., 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')",
    )
    table_transformer_model_path: Optional[str] = Field(
        default=None,
        description="Path to Table Transformer model for table structure detection",
    )
    yolo_layout_model_name: str = Field(
        default="models/doclayout_yolo_docstructbench_imgsz1024.pt",
        description="YOLO model for document layout detection (DocLayout-YOLO)",
    )
    yolo_layout_confidence: float = Field(
        default=0.3,
        description="Confidence threshold for YOLO layout detection",
    )
    sam_checkpoint_path: Optional[str] = Field(
        default=None,
        description="Local path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth)",
    )

    # Text Comparison Thresholds
    text_similarity_threshold: float = Field(
        default=0.82,
        description="Cosine similarity threshold for considering text unchanged (0.0-1.0)",
    )
    semantic_change_threshold: float = Field(
        default=0.5,
        description="Similarity threshold below which text is considered semantically changed",
    )
    
    # Formatting Comparison Thresholds
    formatting_change_threshold: float = Field(
        default=0.1,
        description="Relative difference threshold for font/spacing changes (0.0-1.0)",
    )
    skip_formatting_for_ocr: bool = Field(
        default=True,
        description="Skip formatting comparison when OCR is used (OCR styles are often unreliable/synthetic)",
    )
    font_size_change_threshold_pt: float = Field(
        default=1.0,
        description="Minimum font size difference in points to detect as change",
    )
    color_difference_threshold: int = Field(
        default=10,
        description="RGB color difference threshold for detecting color changes",
    )
    spacing_change_threshold_ratio: float = Field(
        default=0.1,
        description="Relative spacing change threshold (10% difference)",
    )
    
    # Layout Comparison Thresholds
    page_size_change_threshold: float = Field(
        default=0.05,
        description="Relative page dimension change threshold (5% difference)",
    )
    block_alignment_distance_threshold: float = Field(
        default=100.0,
        description="Maximum distance in points for block alignment matching",
    )
    page_alignment_confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence for page alignment matching",
    )
    
    # Table Comparison Thresholds
    table_overlap_threshold: float = Field(
        default=0.3,
        description="Minimum overlap score (0.0-1.0) for matching tables between pages",
    )
    table_structure_confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence for table structure detection",
    )
    
    # Figure Comparison Thresholds
    figure_overlap_threshold: float = Field(
        default=0.3,
        description="Minimum overlap score (0.0-1.0) for matching figures between pages",
    )
    caption_search_margin: float = Field(
        default=50.0,
        description="Margin in points for searching figure captions near figures",
    )
    caption_search_distance: float = Field(
        default=100.0,
        description="Maximum distance in points below figure to search for captions",
    )
    
    # Header/Footer Detection Thresholds
    header_region_height_ratio: float = Field(
        default=0.1,
        description="Ratio of page height for header region (top 10%)",
    )
    footer_region_height_ratio: float = Field(
        default=0.1,
        description="Ratio of page height for footer region (bottom 10%)",
    )
    header_footer_repetition_threshold: float = Field(
        default=0.33,
        description="Minimum ratio of pages where header/footer must appear to be considered repeating",
    )
    header_footer_match_threshold: float = Field(
        default=0.3,
        description="Minimum similarity score for matching headers/footers between pages",
    )
    
    # Visual Diff Thresholds
    visual_diff_pixel_threshold: int = Field(
        default=30,
        description="Pixel difference threshold for visual diff noise filtering (0-255)",
    )
    visual_diff_dpi: int = Field(
        default=150,
        description="DPI for visual diff rendering (lower = faster)",
    )
    heatmap_overlay_alpha: float = Field(
        default=0.4,
        description="Transparency for heatmap overlay (0.0-1.0)",
    )
    
    # Alignment Thresholds
    page_similarity_search_window: int = Field(
        default=3,
        description="Search window size for page similarity matching (±3 pages)",
    )
    block_alignment_distance_limit: float = Field(
        default=100.0,
        description="Maximum distance in points for block alignment",
    )
    
    # Evaluation Metrics Targets
    target_precision: float = Field(
        default=0.85,
        description="Target precision for change detection (85%)",
    )
    target_recall: float = Field(
        default=0.80,
        description="Target recall for change detection (80%)",
    )
    target_f1_score: float = Field(
        default=0.82,
        description="Target F1 score for change detection (82%)",
    )
    target_alignment_accuracy: float = Field(
        default=0.95,
        description="Target alignment accuracy (95%)",
    )

    # OCR Settings
    use_ocr_for_all_documents: bool = Field(
        default=False,
        description="Use DeepSeek-OCR for all documents (not just scanned). Enables OCR-enhanced comparison.",
    )
    ocr_enhancement_mode: str = Field(
        default="auto",
        description="OCR enhancement mode: 'auto' (only scanned), 'hybrid' (OCR + native), 'ocr_only' (force OCR for all)",
    )
    ocr_engine: str = Field(
        default="paddle",
        description="OCR engine to use. Options: 'deepseek' (requires GPU/MPS), 'paddle' (fast, accurate), 'tesseract' (legacy).",
    )
    paddle_ocr_lang: str = Field(
        default="en",
        description="PaddleOCR language code (e.g., 'en', 'lt', 'zh')",
    )
    tesseract_lang: str = Field(
        default="eng",
        description="Tesseract language code (e.g., 'eng', 'lit', 'chi_sim')",
    )
    min_ocr_blocks_per_page: int = Field(
        default=3,
        description="Minimum blocks per page to consider OCR successful. Below this threshold triggers fallback to next engine.",
    )
    
    # DeepSeek-OCR Inference Parameters
    deepseek_base_size: int = Field(
        default=1024,
        description="Base size for DeepSeek-OCR high-res processing (1024 for standard, 1280 for large)",
    )
    deepseek_image_size: int = Field(
        default=640,
        description="Image size for DeepSeek-OCR processing (640 for standard, 1024 for base)",
    )
    deepseek_crop_mode: bool = Field(
        default=True,
        description="Enable crop mode for DeepSeek-OCR (True for Gundam mode, False for standard)",
    )
    deepseek_grounding_enabled: bool = Field(
        default=True,
        description="Enable grounding mode to get bounding boxes with Markdown output",
    )

    # Performance
    max_pages: int = Field(default=60, description="Maximum pages to process per document")
    seconds_per_page_target: float = Field(default=3.0, description="Performance target: <3s per page")
    num_workers: int = Field(default=4, description="Parallel workers for OCR/alignment")
    use_gpu: bool = Field(default=False, description="Enable GPU acceleration if available")
    batch_size: int = Field(default=32, description="Batch size for embedding computation")

    # UI
    ui_theme: str = Field(default="light", description="Default UI theme")
    render_dpi: int = Field(default=144, description="DPI for PDF page rendering in UI")
    render_scale_factor: float = Field(default=2.0, description="Scale factor for high-DPI rendering")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Return a cached settings instance."""
    return _get_settings()


@lru_cache()
def _get_settings() -> Settings:
    return Settings()


settings = get_settings()
