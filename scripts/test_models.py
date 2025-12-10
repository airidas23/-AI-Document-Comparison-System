"""Test that downloaded models can be loaded and used."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging import configure_logging, logger

configure_logging()


def test_deepseek_ocr(model_path: str) -> bool:
    """Test DeepSeek-OCR model loading."""
    logger.info("Testing DeepSeek-OCR model at: %s", model_path)
    
    try:
        from transformers import AutoModel, AutoProcessor
        import torch
        
        model_dir = Path(model_path)
        if not model_dir.exists():
            logger.error("Model directory does not exist: %s", model_dir)
            return False
        
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        logger.info("✓ Processor loaded successfully")
        
        logger.info("Loading model...")
        try:
            model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
            logger.info("✓ Model loaded successfully")
        except Exception as model_load_error:
            # Check if it's a transformers version compatibility issue
            if "LlamaFlashAttention2" in str(model_load_error) or "cannot import name" in str(model_load_error):
                logger.warning(
                    "Model loading failed due to transformers version compatibility. "
                    "This may work at runtime. Error: %s", model_load_error
                )
                logger.info("Attempting to load with ignore_mismatched_sizes...")
                try:
                    model = AutoModel.from_pretrained(
                        str(model_dir),
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True,
                    )
                    logger.info("✓ Model loaded with workaround")
                except Exception:
                    logger.error("Could not load model even with workaround")
                    return False
            else:
                raise
        
        # Set to eval mode
        model.eval()
        logger.info("Model set to evaluation mode")
        
        # Check if GPU is available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("✓ Model moved to GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Metal) available but not used for DeepSeek-OCR")
        else:
            logger.info("Using CPU (GPU not available)")
        
        logger.info("✓ DeepSeek-OCR model test PASSED")
        return True
        
    except Exception as exc:
        logger.error("✗ DeepSeek-OCR model test FAILED: %s", exc)
        return False


def test_sentence_transformer(model_path: str) -> bool:
    """Test Sentence Transformer model loading."""
    logger.info("Testing Sentence Transformer model at: %s", model_path)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_dir = Path(model_path)
        if not model_dir.exists():
            logger.error("Model directory does not exist: %s", model_dir)
            return False
        
        logger.info("Loading model...")
        model = SentenceTransformer(str(model_dir))
        logger.info("✓ Model loaded successfully")
        
        # Test encoding
        logger.info("Testing model encoding...")
        test_text = "This is a test sentence."
        embedding = model.encode(test_text, convert_to_tensor=False, show_progress_bar=False)
        logger.info("✓ Encoding test successful (embedding shape: %s)", embedding.shape)
        
        # Test similarity
        from sentence_transformers import util
        text1 = "The cat sat on the mat"
        text2 = "A cat was sitting on the mat"
        emb1 = model.encode(text1, convert_to_tensor=True, show_progress_bar=False)
        emb2 = model.encode(text2, convert_to_tensor=True, show_progress_bar=False)
        similarity = util.cos_sim(emb1, emb2).item()
        logger.info("✓ Similarity test successful (similarity: %.3f)", similarity)
        
        logger.info("✓ Sentence Transformer model test PASSED")
        return True
        
    except Exception as exc:
        logger.error("✗ Sentence Transformer model test FAILED: %s", exc)
        return False


def main() -> None:
    """Test all models."""
    from config.settings import settings
    
    logger.info("=" * 60)
    logger.info("Model Verification Test")
    logger.info("=" * 60)
    logger.info("")
    
    results = {}
    
    # Test DeepSeek-OCR
    logger.info("Testing DeepSeek-OCR...")
    logger.info("-" * 60)
    results['deepseek'] = test_deepseek_ocr(settings.deepseek_ocr_model_path)
    logger.info("")
    
    # Test Sentence Transformer
    logger.info("Testing Sentence Transformer...")
    logger.info("-" * 60)
    results['sentence_transformer'] = test_sentence_transformer(settings.sentence_transformer_model)
    logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    all_passed = all(results.values())
    
    for model_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info("%s: %s", model_name.replace('_', ' ').title(), status)
    
    logger.info("")
    if all_passed:
        logger.info("✓ All models are working correctly!")
        sys.exit(0)
    else:
        logger.error("✗ Some models failed to load. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

