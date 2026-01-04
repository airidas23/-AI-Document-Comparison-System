#!/usr/bin/env python3
"""Quick script to download the correct DeepSeek-OCR model."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.setup_models import download_model, validate_model_type
from utils.logging import configure_logging, logger

configure_logging()


def main() -> None:
    """Download DeepSeek-OCR model and validate it."""
    models_dir = project_root / "models"
    deepseek_path = models_dir / "deepseek-ocr"
    
    logger.info("=" * 60)
    logger.info("Downloading DeepSeek-OCR model...")
    logger.info("=" * 60)
    
    success = download_model(
        model_id="deepseek-ai/deepseek-ocr",
        local_dir=deepseek_path,
        model_name="DeepSeek-OCR",
        expected_model_type="deepseekocr",
    )
    
    if success:
        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ DeepSeek-OCR model downloaded and validated!")
        logger.info("")
        logger.info("Verifying model type...")
        
        # Final verification
        if validate_model_type(deepseek_path, "deepseekocr"):
            logger.info("")
            logger.info("✓ Model type verification passed!")
            logger.info("")
            logger.info("You can now use the model with:")
            logger.info("  python -c \"from transformers import AutoConfig; c=AutoConfig.from_pretrained('models/deepseek-ocr', trust_remote_code=True); print(f'Model type: {c.model_type}')\"")
            sys.exit(0)
        else:
            logger.error("Model type verification failed!")
            sys.exit(1)
    else:
        logger.error("Failed to download DeepSeek-OCR model!")
        sys.exit(1)


if __name__ == "__main__":
    main()

