"""Download and setup models for the AI Document Comparison System."""
from __future__ import annotations

import sys
import shutil
import urllib.request
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging import configure_logging, logger

configure_logging()


def validate_model_type(local_dir: Path, expected_model_type: str = None) -> bool:
    """
    Validate that the model at local_dir is the correct type.
    
    For DeepSeek-OCR, we check for 'DeepseekOCR' in architectures or auto_map,
    not model_type (which is 'deepseek_vl_v2' because OCR uses VL2 architecture).
    
    Args:
        local_dir: Local directory path to the model
        expected_model_type: Expected model type identifier:
            - 'deepseekocr': Check for DeepseekOCRForCausalLM architecture
    
    Returns:
        True if model type matches (or if expected_model_type is None), False otherwise
    """
    if expected_model_type is None:
        return True  # Skip validation if no expected type provided
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(str(local_dir), trust_remote_code=True)
        
        if expected_model_type.lower() == 'deepseekocr':
            # DeepSeek-OCR uses deepseek_vl_v2 as base, but has DeepseekOCR architecture
            architectures = getattr(config, 'architectures', [])
            auto_map = getattr(config, 'auto_map', {})
            
            # Check if it has DeepseekOCR architecture
            has_ocr_arch = any('DeepseekOCR' in arch for arch in architectures)
            has_ocr_auto = any('DeepseekOCR' in str(v) for v in auto_map.values())
            
            if has_ocr_arch or has_ocr_auto:
                logger.info(
                    "✓ Model validated as DeepSeek-OCR (architectures: %s, auto_map: %s)",
                    architectures,
                    list(auto_map.values()) if auto_map else []
                )
                return True
            else:
                logger.error(
                    "Model at %s does not appear to be DeepSeek-OCR. "
                    "Architectures: %s, Auto map: %s",
                    local_dir,
                    architectures,
                    auto_map
                )
                return False
        else:
            # For other model types, check model_type directly
            actual_type = getattr(config, 'model_type', None)
            if actual_type is None:
                logger.warning("Could not determine model_type from config at %s", local_dir)
                return True
            
            actual_type_lower = actual_type.lower()
            expected_type_lower = expected_model_type.lower()
            
            if actual_type_lower != expected_type_lower:
                logger.error(
                    "Model type mismatch! Expected '%s', but found '%s' at %s",
                    expected_model_type,
                    actual_type,
                    local_dir
                )
                return False
            
            logger.info("✓ Model type validated: %s", actual_type)
            return True
    except Exception as exc:
        logger.warning("Could not validate model type: %s", exc)
        return True  # Don't fail download if validation fails


def download_model(
    model_id: str,
    local_dir: Path,
    model_name: str,
    expected_model_type: str = None,
) -> bool:
    """
    Download a model from HuggingFace to a local directory.
    
    Args:
        model_id: HuggingFace model identifier (e.g., 'deepseek-ai/deepseek-ocr')
        local_dir: Local directory path to save the model
        model_name: Human-readable model name for logging
        expected_model_type: Expected model_type to validate after download (e.g., 'deepseekocr')
    
    Returns:
        True if download successful, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub is required. Install via: pip install huggingface_hub"
        )
        return False
    
    # Check if model already exists
    if local_dir.exists() and any(local_dir.iterdir()):
        logger.info("%s already exists at %s, validating...", model_name, local_dir)
        # Validate existing model
        if expected_model_type and not validate_model_type(local_dir, expected_model_type):
            logger.error(
                "Existing model at %s has wrong type. Please remove it and re-run this script.",
                local_dir
            )
            return False
        logger.info("✓ Existing model validated, skipping download")
        return True
    
    logger.info("Downloading %s from HuggingFace...", model_name)
    logger.info("Model ID: %s", model_id)
    logger.info("Target directory: %s", local_dir)
    
    try:
        # Create parent directory if it doesn't exist
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Download model
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        
        # Verify download
        if local_dir.exists() and any(local_dir.iterdir()):
            logger.info("✓ Successfully downloaded %s to %s", model_name, local_dir)
            
            # Validate model type if expected type provided
            if expected_model_type:
                if not validate_model_type(local_dir, expected_model_type):
                    logger.error(
                        "Downloaded model has wrong type! This may indicate a corrupted download."
                    )
                    return False
            
            return True
        else:
            logger.error("Download completed but model files not found at %s", local_dir)
            return False
            
    except Exception as exc:
        logger.error("Failed to download %s: %s", model_name, exc)
        return False


def download_file(url: str, target_path: Path, name: str) -> bool:
    """Download a single file to a target path."""
    if target_path.exists():
        logger.info("%s already exists at %s, skipping download", name, target_path)
        return True
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s from %s", name, url)
        urllib.request.urlretrieve(url, target_path)
        logger.info("✓ Saved %s to %s", name, target_path)
        return True
    except Exception as exc:
        logger.error("Failed to download %s: %s", name, exc)
        return False


def download_yolo_layout_model(models_dir: Path) -> bool:
    """Download YOLOv11x weights for layout detection."""
    model_path = models_dir / "yolo11x.pt"
    if model_path.exists():
        logger.info("YOLOv11x already present at %s", model_path)
        return True

    # Try huggingface_hub for reproducible fetch
    try:
        from huggingface_hub import hf_hub_download

        logger.info("Downloading YOLOv11x via huggingface_hub...")
        tmp_path = hf_hub_download(
            repo_id="ultralytics/YOLOv11",
            filename="yolov11x.pt",
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path, model_path)
        logger.info("✓ YOLOv11x downloaded to %s", model_path)
        return True
    except Exception as exc:
        logger.warning("huggingface_hub download failed for YOLOv11x: %s", exc)

    # Fallback direct URL
    yolo_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt"
    return download_file(yolo_url, model_path, "YOLOv11x")


def download_sam_checkpoint(models_dir: Path) -> bool:
    """Download SAM ViT-H checkpoint."""
    checkpoint_path = models_dir / "sam_vit_h_4b8939.pth"
    if checkpoint_path.exists():
        logger.info("SAM checkpoint already present at %s", checkpoint_path)
        return True

    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    return download_file(sam_url, checkpoint_path, "SAM ViT-H")


def main() -> None:
    """Download all required models."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    logger.info("Starting model download process...")
    logger.info("Models will be saved to: %s", models_dir)
    
    # Download DeepSeek-OCR model
    deepseek_path = models_dir / "deepseek-ocr"
    deepseek_success = download_model(
        model_id="deepseek-ai/deepseek-ocr",
        local_dir=deepseek_path,
        model_name="DeepSeek-OCR",
        expected_model_type="deepseekocr",  # Validate it's actually DeepSeek-OCR, not VL2
    )
    
    # Download sentence transformer model
    sentence_transformer_path = models_dir / "all-MiniLM-L6-v2"
    sentence_transformer_success = download_model(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=sentence_transformer_path,
        model_name="Sentence Transformer (all-MiniLM-L6-v2)",
    )

    # Download YOLOv11 layout model (disabled as not used)
    # yolo_success = download_yolo_layout_model(models_dir)
    yolo_success = True

    # Download SAM checkpoint (optional)
    sam_success = download_sam_checkpoint(models_dir)
    
    # Summary
    logger.info("=" * 60)
    if deepseek_success and sentence_transformer_success and yolo_success and sam_success:
        logger.info("✓ All models downloaded successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Copy .env.example to .env")
        logger.info("2. Update .env with model paths if needed")
        logger.info("3. Run the application: python app.py")
    else:
        logger.warning("Some models failed to download:")
        if not deepseek_success:
            logger.warning("  ✗ DeepSeek-OCR")
        if not sentence_transformer_success:
            logger.warning("  ✗ Sentence Transformer")
        if not yolo_success:
            logger.warning("  ✗ YOLOv11x")
        if not sam_success:
            logger.warning("  ✗ SAM ViT-H")
        logger.info("")
        logger.info("You can retry the download or download models manually.")
        logger.info("See models/README.md for manual download instructions.")
        sys.exit(1)


if __name__ == "__main__":
    main()

