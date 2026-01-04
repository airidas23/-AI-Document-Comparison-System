"""Update DeepSeek-OCR model to the latest version from HuggingFace."""
from __future__ import annotations

import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging import configure_logging, logger

configure_logging()


def update_deepseek_ocr_model(force: bool = True) -> bool:
    """
    Download/update DeepSeek-OCR model from HuggingFace.
    
    Args:
        force: If True, force re-download even if model exists
        
    Returns:
        True if update successful, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub is required. Install via: pip install huggingface_hub"
        )
        return False
    
    model_id = "deepseek-ai/DeepSeek-OCR"
    models_dir = project_root / "models"
    local_dir = models_dir / "deepseek-ocr"
    
    logger.info("Updating DeepSeek-OCR model...")
    logger.info("Model ID: %s", model_id)
    logger.info("Target directory: %s", local_dir)
    
    # Create backup if model exists and we're forcing update
    backup_dir = None
    if force and local_dir.exists() and any(local_dir.iterdir()):
        backup_dir = models_dir / "deepseek-ocr.backup"
        if backup_dir.exists():
            logger.info("Removing old backup...")
            shutil.rmtree(backup_dir)
        logger.info("Creating backup of existing model...")
        shutil.copytree(local_dir, backup_dir)
        logger.info("Backup created at: %s", backup_dir)
    
    try:
        # Create parent directory if it doesn't exist
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # If forcing update, remove existing directory first
        if force and local_dir.exists():
            logger.info("Removing existing model directory...")
            shutil.rmtree(local_dir)
        
        # Download model (this will get the latest version)
        logger.info("Downloading latest version from HuggingFace...")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
        )
        
        # Verify download
        if local_dir.exists() and any(local_dir.iterdir()):
            logger.info("✓ Successfully updated DeepSeek-OCR to latest version at %s", local_dir)
            
            # Clean up backup if update succeeded
            if backup_dir and backup_dir.exists():
                logger.info("Removing backup (update successful)...")
                shutil.rmtree(backup_dir)
            
            return True
        else:
            logger.error("Download completed but model files not found at %s", local_dir)
            
            # Restore backup if update failed
            if backup_dir and backup_dir.exists():
                logger.warning("Restoring backup due to failed update...")
                if local_dir.exists():
                    shutil.rmtree(local_dir)
                shutil.move(backup_dir, local_dir)
                logger.info("Backup restored to %s", local_dir)
            
            return False
            
    except Exception as exc:
        logger.error("Failed to update DeepSeek-OCR: %s", exc)
        
        # Restore backup if update failed
        if backup_dir and backup_dir.exists():
            logger.warning("Restoring backup due to error...")
            if local_dir.exists():
                shutil.rmtree(local_dir)
            shutil.move(backup_dir, local_dir)
            logger.info("Backup restored to %s", local_dir)
        
        return False


def main() -> None:
    """Update DeepSeek-OCR model to latest version."""
    logger.info("=" * 60)
    logger.info("DeepSeek-OCR Model Update")
    logger.info("=" * 60)
    
    success = update_deepseek_ocr_model(force=True)
    
    logger.info("=" * 60)
    if success:
        logger.info("✓ DeepSeek-OCR model updated successfully!")
        sys.exit(0)
    else:
        logger.error("✗ Failed to update DeepSeek-OCR model")
        sys.exit(1)


if __name__ == "__main__":
    main()

