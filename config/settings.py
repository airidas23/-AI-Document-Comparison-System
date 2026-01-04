"""Configuration management for model paths, thresholds, and performance settings."""
from __future__ import annotations

from functools import lru_cache

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # Fallback for Pydantic v1
    from pydantic import BaseSettings  # type: ignore
    SettingsConfigDict = None  # type: ignore

from typing import List, Optional, Literal

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
    table_cell_text_threshold: float = Field(
        default=0.85,
        description="Minimum text similarity for cell-level table comparison (0.0-1.0)",
    )
    table_border_detection_enabled: bool = Field(
        default=True,
        description="Enable border change detection using drawing signature analysis",
    )
    table_style_inset_change_threshold: float = Field(
        default=0.02,
        description=(
            "Minimum absolute change (0.0-1.0, relative to table width/height) in table content insets "
            "to report a table style change (padding/border weight)."
        ),
    )
    table_style_bbox_change_threshold: float = Field(
        default=0.01,
        description=(
            "Minimum absolute change (0.0-1.0, relative to page) in normalized table bbox coordinates "
            "to report a table style change when text is unchanged."
        ),
    )
    
    # Figure Comparison Thresholds
    figure_overlap_threshold: float = Field(
        default=0.3,
        description="Minimum overlap score (0.0-1.0) for matching figures between pages",
    )
    figure_hash_threshold: int = Field(
        default=8,
        description="Maximum Hamming distance for perceptual hash to consider figures identical (0-64)",
    )
    figure_caption_threshold: float = Field(
        default=0.80,
        description="Minimum caption similarity for figure matching (0.0-1.0)",
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
    
    # Layout Change Classification
    layout_position_tolerance: float = Field(
        default=0.05,
        description="Position change tolerance as ratio of page size (5% = ~30pt on standard page)",
    )
    layout_size_tolerance: float = Field(
        default=0.10,
        description="Size change tolerance as ratio (10% change triggers resize detection)",
    )
    
    # Formula Comparison
    formula_latex_threshold: float = Field(
        default=0.90,
        description="Minimum LaTeX similarity for formula matching (0.0-1.0)",
    )
    formula_ocr_engine: str = Field(
        default="pix2tex",
        description="OCR engine for formula extraction: 'pix2tex' (neural LaTeX OCR) or 'visual' (fallback)",
    )
    
    # Debug Mode
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode for detailed JSON exports and intermediate results",
    )
    debug_output_path: str = Field(
        default="./debug_output",
        description="Directory path for debug output files",
    )

    # OCR / scanned alignment tuning
    ocr_min_text_similarity_for_match: float = Field(
        default=0.7,
        description=(
            "Minimum text similarity gate for matching OCR-extracted items. "
            "If similarity is below this, items are not matched and will be treated as added/deleted."
        ),
    )
    ocr_layout_tolerance_ratio: float = Field(
        default=0.03,
        description=(
            "Relative bbox tolerance for layout shifts on OCR/scanned pages (e.g. 0.03 = 3% of page). "
            "Scanned documents often have small global drift/scale." 
        ),
    )
    ocr_translation_estimation_min_similarity: float = Field(
        default=0.85,
        description=(
            "Minimum text similarity for a block pair to be considered when estimating a global page translation (dx/dy) on OCR pages."
        ),
    )
    
    # OCR-specific comparison thresholds (calibrated for reducing false positives)
    ocr_text_similarity_threshold: float = Field(
        default=0.72,
        description=(
            "OCR matching is noisier (line breaks, spacing). Lower threshold = fewer false diffs. "
            "Used instead of text_similarity_threshold when extraction_method contains 'ocr'."
        ),
    )
    ocr_min_change_chars: int = Field(
        default=1,
        description=(
            "Ignore microscopic OCR noise unless >= this many real character edits. "
            "Set to 1 to detect single-char typos (e.g. rates->raates)."
        ),
    )
    ocr_min_change_ratio: float = Field(
        default=0.005,
        description=(
            "Ignore tiny diffs: changed_chars / max(len(before), len(after)) must exceed this. "
            "Set to 0.005 (0.5%) to catch 1-char typos in ~200-char paragraphs."
        ),
    )
    ocr_ignore_punctuation_diffs: bool = Field(
        default=True,
        description=(
            "Filter out punctuation-only diffs on OCR pages. "
            "OCR engines often misread punctuation (period/comma, dash variants)."
        ),
    )
    ocr_ignore_whitespace_diffs: bool = Field(
        default=True,
        description=(
            "Filter out whitespace-only diffs on OCR pages. "
            "OCR spacing is synthetic and varies between engines/DPI."
        ),
    )
    ocr_ignore_case_diffs: bool = Field(
        default=True,
        description=(
            "Filter out case-only diffs on OCR pages. "
            "OCR sometimes confuses uppercase/lowercase in certain fonts."
        ),
    )
    ocr_aggressive_noise_filter: bool = Field(
        default=True,
        description=(
            "Enable aggressive post-classification OCR noise filtering. "
            "Removes diffs that are likely OCR artifacts based on multiple heuristics."
        ),
    )
    skip_header_footer_for_ocr: bool = Field(
        default=True,
        description=(
            "Header/footer detection is unstable on OCR; skip for prototype. "
            "Set False once you have stable region detection."
        ),
    )
    skip_layout_comparison_for_ocr: bool = Field(
        default=True,
        description=(
            "Skip fine-grained layout comparison (spacing, line shifts) for OCR pages. "
            "OCR bbox positions have natural variance from scan skew/DPI rounding."
        ),
    )
    ocr_layout_block_shift_only: bool = Field(
        default=True,
        description=(
            "For OCR pages, only report layout changes for large block-level shifts. "
            "Ignores line-level spacing differences which are noisy for scanned docs."
        ),
    )
    ocr_layout_position_tolerance: float = Field(
        default=0.05,
        description=(
            "Position tolerance for OCR layout comparison (5% = ~30pt on standard page). "
            "Higher than digital PDF tolerance due to scan drift."
        ),
    )
    ocr_table_cell_text_threshold: float = Field(
        default=0.75,
        description=(
            "Cell similarity threshold for OCR tables (lower than digital). "
            "Used for cell-level table comparison when OCR is involved."
        ),
    )
    ocr_table_structure_confidence_threshold: float = Field(
        default=0.75,
        description=(
            "Only do OCR table cell-level compare when structure confidence is high. "
            "Below this, report 'table_region_changed' instead of noisy cell diffs."
        ),
    )
    ocr_paragraph_merge_enabled: bool = Field(
        default=True,
        description=(
            "Merge OCR lines into paragraphs for A-phase comparison. "
            "Reduces false positives from line-break differences."
        ),
    )
    ocr_paragraph_gap_threshold: float = Field(
        default=1.5,
        description=(
            "Vertical gap multiplier (relative to line height) to detect paragraph breaks. "
            "Lines with gaps larger than line_height * this value start a new paragraph."
        ),
    )
    line_reflow_paragraph_merge: bool = Field(
        default=True,
        description=(
            "Also merge lines into paragraphs for digital PDFs when font size changes may cause line reflow. "
            "This reduces false positive character_change diffs when text is identical but line breaks moved."
        ),
    )
    line_reflow_font_size_diff_threshold: float = Field(
        default=0.2,
        description=(
            "Minimum font size difference (in points) between pages to trigger paragraph merge. "
            "When pages have font size differences >= this value, paragraph merge is applied."
        ),
    )
    always_merge_paragraphs_for_comparison: bool = Field(
        default=True,
        description=(
            "Always merge lines into paragraphs before comparison for digital PDFs. "
            "This helps reduce false positives when text content changes cause line reflow, "
            "even without font size changes (e.g., adding words causes text to wrap differently)."
        ),
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

    ocr_fallback_enabled: bool = Field(
        default=True,
        description="If True, allows fallback to other OCR engines if the primary one fails. If False, only the selected engine is used.",
    )

    # Scanned OCR routing policy
    # - strict: user-selected engine only (fail fast, no fallback)
    # - auto_fallback: try a chain of engines (deterministic order)
    ocr_scanned_policy: Literal["strict", "auto_fallback"] = Field(
        default="auto_fallback",  # Demo mode: fallback so one slow page doesn't kill run
        description=(
            "Scanned OCR policy. 'strict' = fail fast with the selected engine only; "
            "'auto_fallback' = try a configured engine chain on failures/timeouts."
        ),
    )
    ocr_scanned_fallback_chain: List[str] = Field(
        default_factory=lambda: ["paddle", "tesseract"],
        description=(
            "Fallback chain used only in auto_fallback mode when the first engine fails. "
            "Does not have to include 'deepseek'."
        ),
    )
    ocr_scanned_default_chain: List[str] = Field(
        default_factory=lambda: ["paddle", "tesseract", "deepseek"],
        description=(
            "Default engine chain used when the user selects AUTO for scanned PDFs. "
            "Order is deterministic."
        ),
    )

    ocr_engine_priority: List[str] = Field(
        default_factory=lambda: ["paddle", "tesseract"],
        description=(
            "Default OCR engine priority list for multi-engine fallback. "
            "Engines are attempted in order and skipped if dependencies/hardware are unavailable. "
            "DeepSeek is NOT included by default due to slow initialization (~30s) and inference (~2min). "
            "Add 'deepseek' to this list explicitly if you want to use it as fallback."
        ),
    )
    paddle_ocr_lang: str = Field(
        default="en",
        description="PaddleOCR language code (e.g., 'en', 'lt', 'zh')",
    )
    paddle_render_dpi: int = Field(
        default=100,
        description="DPI for rendering pages before PaddleOCR (lower=faster, 100 is good balance)",
    )

    paddle_retry_with_textline_orientation: bool = Field(
        default=True,
        description=(
            "If True, PaddleOCR will retry a page with textline orientation enabled when the first pass "
            "looks low-quality (low confidence / gibberish)."
        ),
    )
    paddle_device: Optional[str] = Field(
        default=None,
        description="PaddleOCR device override, e.g. 'cpu', 'gpu', 'gpu:0'. If None, uses use_gpu flag.",
    )
    paddle_cpu_threads: Optional[int] = Field(
        default=None,
        description="CPU threads for PaddleOCR inference. If None, chooses a sane default.",
    )
    paddle_enable_mkldnn: Optional[bool] = Field(
        default=None,
        description="Enable MKL-DNN acceleration on x86 CPUs. If None, auto-detect.",
    )
    paddle_text_rec_score_thresh: float = Field(
        default=0.0,
        description="Minimum recognition confidence for PaddleOCR results (0.0 keeps all).",
    )
    paddle_return_word_box: bool = Field(
        default=False,
        description="Whether to return word-level boxes (if supported by the current pipeline).",
    )
    tesseract_lang: str = Field(
        default="eng",
        description="Tesseract language code (e.g., 'eng', 'lit', 'chi_sim')",
    )

    tesseract_render_dpi: int = Field(
        default=300,
        description="DPI for rendering pages before Tesseract OCR (lower=faster, 100 is good balance)",
    )
    tesseract_psm_mode: int = Field(
        default=3,
        description="Tesseract page segmentation mode (0-13).",
    )
    tesseract_oem_mode: int = Field(
        default=3,
        description="Tesseract OCR engine mode (0-3).",
    )
    tesseract_min_confidence: int = Field(
        default=30,
        description="Minimum Tesseract confidence (0-100) to keep a text block.",
    )
    tesseract_granularity: str = Field(
        default="word",
        description="Tesseract output granularity: 'word', 'line', or 'block'.",
    )
    tesseract_config_string: str = Field(
        default="-c preserve_interword_spaces=1",
        description="Additional Tesseract config string appended to --psm/--oem. Default enables preserve_interword_spaces for better column/table handling.",
    )

    # Tesseract debugging
    tesseract_debug_min_chars: int = Field(
        default=50,
        description=(
            "If a page produces fewer characters than this threshold, "
            "the preprocessed render may be saved to debug_output_path for inspection."
        ),
    )
    tesseract_debug_save_low_text_images: bool = Field(
        default=True,
        description=(
            "If True, save preprocessed page images when OCR output looks sparse. "
            "Useful for diagnosing scan quality, DPI, or preprocessing issues."
        ),
    )

    # Tesseract typo-sensitive options
    tesseract_disable_dawg: bool = Field(
        default=False,
        description=(
            "If True, disables Tesseract system/frequency dictionaries (DAWG). "
            "This can reduce bias toward 'real' words and help preserve typos like rates->raates."
        ),
    )

    # Tesseract PSM fallback
    tesseract_psm_fallback_enabled: bool = Field(
        default=True,
        description="If True, try multiple PSM modes when the first pass returns too little text.",
    )
    tesseract_psm_fallback_order: List[int] = Field(
        default_factory=lambda: [6, 4, 3, 11, 12, 1],
        description=(
            "Ordered PSM modes to try for fallback on sparse/failed OCR. "
            "6=single block, 4=single column, 3=auto, 11/12=sparse text, 1=orientation+script."
        ),
    )
    tesseract_psm_min_chars_ok: int = Field(
        default=200,
        description="If OCR returns at least this many characters, accept the current PSM without further fallback.",
    )

    # Tesseract preprocessing
    tesseract_preprocess_enabled: bool = Field(
        default=True,
        description="If True, apply safe preprocessing (grayscale/autocontrast/denoise) before Tesseract.",
    )
    tesseract_preprocess_binarize: bool = Field(
        default=False,
        description="If True, apply Otsu binarization in preprocessing (can help challenging scans).",
    )
    tesseract_preprocess_unsharp: bool = Field(
        default=True,
        description="If True, apply mild unsharp mask to improve character edges on scans.",
    )
    tesseract_preprocess_median_size: int = Field(
        default=3,
        description="Median filter kernel size used in Tesseract preprocessing (odd integer).",
    )
    tesseract_invert: bool = Field(
        default=False,
        description="If True, invert grayscale image before OCR (useful for white-on-black scans).",
    )
    tesseract_adaptive_psm: bool = Field(
        default=False,
        description="Enable adaptive PSM selection based on detected layout (experimental). If True, may use PSM 4 for multi-column pages.",
    )
    tesseract_adaptive_dpi: bool = Field(
        default=False,
        description="Enable adaptive DPI based on character height. If text is small, may increase DPI to 200-300.",
    )
    min_ocr_blocks_per_page: int = Field(
        default=3,
        description="Minimum blocks per page to consider OCR successful. Below this threshold triggers fallback to next engine.",
    )

    # OCR quality gates (used for retry/fallback decisions and downstream weighting)
    ocr_quality_min_chars_per_page: int = Field(
        default=25,
        description="If OCR extracted fewer chars than this on a page, mark as low quality.",
    )
    ocr_quality_min_avg_confidence: float = Field(
        default=0.55,
        description="If average OCR confidence is below this (0-1), mark as low quality (when confidence is available).",
    )
    ocr_quality_max_gibberish_ratio: float = Field(
        default=0.35,
        description="If gibberish ratio exceeds this (0-1), mark page as low quality.",
    )

    # Scan detection (digital vs scanned)
    scan_detection_sample_pages: int = Field(
        default=5,
        description="How many initial pages to sample when detecting scanned PDFs.",
    )
    scan_detection_min_good_pages_ratio: float = Field(
        default=0.1,
        description="Minimum ratio of sampled pages with usable text to consider the PDF digital.",
    )
    scan_detection_page_text_min_chars: int = Field(
        default=50,
        description="Minimum characters on a sampled page to consider it as having a text layer.",
    )
    scan_detection_garbage_max_nonprintable_ratio: float = Field(
        default=0.02,
        description="If non-printable character ratio exceeds this, treat extracted text as garbage.",
    )
    scan_detection_garbage_min_alnum_ratio: float = Field(
        default=0.12,
        description="If alphanumeric ratio is below this (and text is long enough), treat extracted text as garbage.",
    )
    hybrid_ocr_min_word_overlap: float = Field(
        default=0.25,
        description="Hybrid mode safety: minimum word-overlap ratio (0-1) required to allow OCR output to replace native text blocks.",
    )
    hybrid_ocr_max_length_ratio: float = Field(
        default=3.0,
        description="Hybrid mode safety: if OCR text length exceeds native text length by this ratio and overlap is low, treat OCR as hallucinated.",
    )
    hybrid_ocr_reject_repetition: bool = Field(
        default=True,
        description="Hybrid mode safety: reject OCR output that looks like repetition/hallucination (recommended for DeepSeek).",
    )
    
    # DeepSeek-OCR Inference Parameters
    deepseek_render_dpi: int = Field(
        # Low-resource default for Apple Silicon: reduces pixels -> faster OCR and lower RAM/VRAM.
        default=60,
        description="DPI for rendering pages before DeepSeek-OCR (lower=faster, lower memory)",
    )
    deepseek_base_size: int = Field(
        # Keep to a known supported mode. 512 is the lightest practical setting.
        default=512,
        description="Base size for DeepSeek-OCR processing (512=fastest supported, 1024=better quality)",
    )
    deepseek_image_size: int = Field(
        # Match base_size for simplest/fastest path (no dynamic patching).
        default=512,
        description="Image size for DeepSeek-OCR processing (512=fastest supported, 640/1024=better quality)",
    )
    deepseek_crop_mode: bool = Field(
        default=False,
        description="Enable crop mode for DeepSeek-OCR. Set to False for faster inference on M-series chips.",
    )
    deepseek_grounding_enabled: bool = Field(
        # Grounding increases work and often slows down generation; disable for low-resource mode.
        default=False,
        description="Enable grounding mode to get bounding boxes with Markdown output (slower)",
    )
    
    # Dynamic OCR / Sanity / Retry
    adaptive_ocr_enabled: bool = Field(
        default=True,
        description="Enable dynamic OCR retries (DPI/image_size) + sanity checks to avoid hallucinated output.",
    )
    deepseek_tmp_image_format: str = Field(
        default="png",
        description="Temp image format for DeepSeek infer(): 'png' or 'jpeg'.",
    )
    deepseek_tmp_jpeg_quality: int = Field(
        default=92,
        description="If deepseek_tmp_image_format='jpeg', this quality is used.",
    )
    deepseek_min_chars_quick: int = Field(
        default=500,
        description="Minimum characters for accepting DeepSeek OCR output in quick pass.",
    )
    deepseek_max_url_like_ratio: float = Field(
        default=0.01,
        description="Reject OCR if it looks like web/markdown junk (url-like tokens ratio too high).",
    )
    deepseek_max_nonprintable_ratio: float = Field(
        default=0.02,
        description="Reject OCR if non-printable chars ratio too high (garbage).",
    )
    deepseek_min_alnum_ratio: float = Field(
        default=0.12,
        description="Reject OCR if alnum ratio too low (garbage).",
    )
    deepseek_retry_dpi: int = Field(
        default=96,
        description="If quick OCR looks bad, rerender at this DPI and retry.",
    )
    deepseek_retry_image_size: int = Field(
        default=768,
        description="If quick OCR looks bad, retry infer with at least this image_size.",
    )
    deepseek_grounding_precision_only: bool = Field(
        default=True,
        description="Keep grounding OFF in quick pass; allow enabling it only in precision/ROI.",
    )

    # ==========================================================================
    # DeepSeek Guardrails (v1) - Timeout, Memory, Subprocess, Academic-Safe
    # ==========================================================================
    deepseek_enabled: bool = Field(
        default=True,
        description="Master switch for DeepSeek engine. If False, always falls back to other engines.",
    )
    deepseek_disable_parallel: bool = Field(
        default=True,
        description="Disable all parallelization in DeepSeek (no loky/ProcessPool). Required to avoid semaphore leaks.",
    )
    deepseek_use_persistent_worker: bool = Field(
        default=True,
        description="Use persistent worker subprocess that keeps model loaded (WebUI-style). Eliminates ~30s model reload per page. If False, falls back to legacy subprocess-per-page.",
    )
    deepseek_hard_timeout: bool = Field(
        default=False,  # Demo mode: reuse global model instance (WebUI-style warm model)
        description="Use subprocess-based hard timeout (killable). If False, uses soft timeout (cooperative).",
    )
    deepseek_timeout_sec_per_page: int = Field(
        default=60,
        description="Maximum seconds per page for DeepSeek inference. On timeout, triggers fallback.",
    )
    deepseek_memory_soft_mb: int = Field(
        default=4500,
        description="Soft memory limit (MB). If RSS exceeds this, attempt graceful abort.",
    )
    deepseek_memory_hard_mb: int = Field(
        default=6000,
        description="Hard memory limit (MB). If RSS exceeds this, force kill subprocess.",
    )
    deepseek_max_pages_per_doc: int = Field(
        default=3,
        description="Maximum pages to process with DeepSeek per document. Others go to fallback engine.",
    )
    deepseek_max_side_px: int = Field(
        default=2000,
        description="Max image side (width or height) in pixels. Larger images are resized before inference.",
    )
    deepseek_max_retries: int = Field(
        default=1,
        description="Maximum retry attempts per mode (0=no retries, 1=one retry). Low value recommended for CPU.",
    )
    
    # DeepSeek Academic-Safe Validation Thresholds
    deepseek_academic_safe_mode: bool = Field(
        default=True,
        description="Enable academic-safe validation (relaxed thresholds for scholarly PDFs with DOI/URLs).",
    )
    deepseek_academic_min_chars: int = Field(
        default=200,
        description="Tier 1 reject: minimum characters for valid output (academic mode).",
    )
    deepseek_academic_min_alnum: float = Field(
        default=0.40,
        description="Tier 1 reject: minimum alphanumeric ratio (academic mode).",
    )
    deepseek_academic_max_nonprintable: float = Field(
        default=0.05,
        description="Tier 1 reject: maximum non-printable character ratio (academic mode).",
    )
    deepseek_academic_url_like_warn: float = Field(
        default=0.03,
        description="Tier 2 warn: URL-like token ratio threshold. Triggers header/footer strip, not hard reject.",
    )
    deepseek_academic_max_url_ratio: float = Field(
        default=0.30,
        description="Tier 2 reject: maximum URL token ratio for academic content.",
    )
    deepseek_academic_max_repetition: float = Field(
        default=0.50,
        description="Tier 1 reject: maximum repetition ratio (model hallucination detection).",
    )
    deepseek_header_footer_strip_pct: float = Field(
        default=0.08,
        description="Header/footer strip zone (percent of page height from top/bottom).",
    )
    
    # DeepSeek Mode Presets (DataCamp-inspired)
    deepseek_modes_priority: List[str] = Field(
        default_factory=lambda: ["plain_text"],
        description="Mode priority for DeepSeek inference. Options: 'plain_text', 'doc_markdown'. Start with 'plain_text' for stability.",
    )
    deepseek_mode_presets: dict = Field(
        default_factory=lambda: {
            "plain_text": {
                "prompt": "<image>\nFree OCR.",
                "base_size": 768,
                "image_size": 512,
                "crop_mode": False,
                "description": "Fast, stable text-only extraction",
            },
            "doc_markdown": {
                "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
                "base_size": 1024,
                "image_size": 640,
                "crop_mode": False,
                "description": "Structured output (headings/tables), slower",
            },
        },
        description="Mode presets for DeepSeek inference with prompt/size configurations.",
    )

    # Performance
    max_pages: int = Field(default=60, description="Maximum pages to process per document")
    seconds_per_page_target: float = Field(default=3.0, description="Performance target: <3s per page")
    num_workers: int = Field(default=2, description="Parallel workers for OCR/alignment")
    use_gpu: bool = Field(default=False, description="Enable GPU acceleration if available")
    batch_size: int = Field(default=32, description="Batch size for embedding computation")

    # UI
    ui_theme: str = Field(default="light", description="Default UI theme")
    render_dpi: int = Field(default=144, description="DPI for PDF page rendering in UI")
    render_scale_factor: float = Field(default=2.0, description="Scale factor for high-DPI rendering")



class _SettingsV1Config:
    env_file = ".env"
    env_file_encoding = "utf-8"


if SettingsConfigDict is not None:
    # Pydantic v2 (pydantic-settings)
    Settings.model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
else:
    # Pydantic v1 fallback
    Settings.Config = _SettingsV1Config


def get_settings() -> Settings:
    """Return a cached settings instance."""
    return _get_settings()


@lru_cache()
def _get_settings() -> Settings:
    return Settings()


settings = get_settings()
