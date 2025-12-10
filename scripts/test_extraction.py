"""Test extraction modules with downloaded models."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging import configure_logging, logger

configure_logging()


def test_settings_loading() -> bool:
    """Test that settings can load model paths."""
    logger.info("Testing settings loading...")
    try:
        from config.settings import settings
        
        logger.info("  DeepSeek-OCR path: %s", settings.deepseek_ocr_model_path)
        logger.info("  Sentence Transformer: %s", settings.sentence_transformer_model)
        
        # Check if paths exist
        deepseek_path = Path(settings.deepseek_ocr_model_path)
        if deepseek_path.exists():
            logger.info("  ✓ DeepSeek-OCR path exists")
        else:
            logger.warning("  ✗ DeepSeek-OCR path does not exist: %s", deepseek_path)
        
        sentence_transformer_path = Path(settings.sentence_transformer_model)
        if sentence_transformer_path.exists():
            logger.info("  ✓ Sentence Transformer path exists")
        else:
            logger.info("  ℹ Sentence Transformer using HuggingFace name (will download on first use)")
        
        logger.info("✓ Settings loading test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ Settings loading test FAILED: %s", exc)
        return False


def test_deepseek_ocr_loading() -> bool:
    """Test that DeepSeek-OCR can be loaded via extraction module."""
    logger.info("Testing DeepSeek-OCR loading via extraction module...")
    try:
        from extraction.ocr_processor import get_ocr_instance
        from config.settings import settings
        
        logger.info("  Getting OCR instance with path: %s", settings.deepseek_ocr_model_path)
        ocr = get_ocr_instance(settings.deepseek_ocr_model_path)
        
        logger.info("  OCR instance created, testing model loading...")
        # Trigger model loading by calling _load_model
        ocr._load_model()
        
        if ocr._model is not None:
            logger.info("  ✓ DeepSeek-OCR model loaded successfully")
            logger.info("  ✓ Processor loaded: %s", ocr._processor is not None)
            logger.info("✓ DeepSeek-OCR loading test PASSED")
            return True
        else:
            logger.error("  ✗ Model failed to load (model is None)")
            return False
            
    except Exception as exc:
        logger.error("✗ DeepSeek-OCR loading test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_sentence_transformer_loading() -> bool:
    """Test that Sentence Transformer can be loaded (used in comparison)."""
    logger.info("Testing Sentence Transformer loading...")
    try:
        from comparison.text_comparison import TextComparator
        from config.settings import settings
        
        logger.info("  Creating TextComparator with model: %s", settings.sentence_transformer_model)
        comparator = TextComparator()
        
        if comparator.model is not None:
            logger.info("  ✓ Sentence Transformer model loaded successfully")
            
            # Test encoding
            test_text = "This is a test"
            embedding = comparator.model.encode(test_text, convert_to_tensor=False, show_progress_bar=False)
            logger.info("  ✓ Encoding test successful (shape: %s)", embedding.shape)
            
            logger.info("✓ Sentence Transformer loading test PASSED")
            return True
        else:
            logger.error("  ✗ Model is None")
            return False
            
    except Exception as exc:
        logger.error("✗ Sentence Transformer loading test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_extraction_module_imports() -> bool:
    """Test that all extraction modules can be imported."""
    logger.info("Testing extraction module imports...")
    try:
        from extraction import extract_pdf
        from extraction.pdf_parser import parse_pdf
        from extraction.ocr_processor import ocr_pdf, get_ocr_instance
        from extraction.layout_analyzer import analyze_layout
        from extraction.header_footer_detector import detect_headers_footers
        
        logger.info("  ✓ extract_pdf imported")
        logger.info("  ✓ parse_pdf imported")
        logger.info("  ✓ ocr_pdf imported")
        logger.info("  ✓ analyze_layout imported")
        logger.info("  ✓ detect_headers_footers imported")
        
        logger.info("✓ Extraction module imports test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ Extraction module imports test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_pdf_parser_basic() -> bool:
    """Test PDF parser can be instantiated (without actual PDF)."""
    logger.info("Testing PDF parser basic functionality...")
    try:
        from extraction.pdf_parser import parse_pdf
        
        # Just test that function exists and has correct signature
        import inspect
        sig = inspect.signature(parse_pdf)
        logger.info("  ✓ parse_pdf signature: %s", sig)
        
        logger.info("✓ PDF parser basic test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ PDF parser basic test FAILED: %s", exc)
        return False


def test_layout_analyzer_basic() -> bool:
    """Test layout analyzer can be instantiated."""
    logger.info("Testing layout analyzer basic functionality...")
    try:
        from extraction.layout_analyzer import analyze_layout
        
        import inspect
        sig = inspect.signature(analyze_layout)
        logger.info("  ✓ analyze_layout signature: %s", sig)
        
        logger.info("✓ Layout analyzer basic test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ Layout analyzer basic test FAILED: %s", exc)
        return False


def test_header_footer_detector_basic() -> bool:
    """Test header/footer detector can be instantiated."""
    logger.info("Testing header/footer detector basic functionality...")
    try:
        from extraction.header_footer_detector import detect_headers_footers
        
        import inspect
        sig = inspect.signature(detect_headers_footers)
        logger.info("  ✓ detect_headers_footers signature: %s", sig)
        
        logger.info("✓ Header/footer detector basic test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ Header/footer detector basic test FAILED: %s", exc)
        return False


def main() -> None:
    """Run all extraction module tests."""
    logger.info("=" * 60)
    logger.info("Extraction Modules Test Suite")
    logger.info("=" * 60)
    logger.info("")
    
    results = {}
    
    # Test 1: Settings loading
    logger.info("Test 1: Settings Loading")
    logger.info("-" * 60)
    results['settings'] = test_settings_loading()
    logger.info("")
    
    # Test 2: Module imports
    logger.info("Test 2: Module Imports")
    logger.info("-" * 60)
    results['imports'] = test_extraction_module_imports()
    logger.info("")
    
    # Test 3: DeepSeek-OCR loading
    logger.info("Test 3: DeepSeek-OCR Model Loading")
    logger.info("-" * 60)
    results['deepseek'] = test_deepseek_ocr_loading()
    logger.info("")
    
    # Test 4: Sentence Transformer loading
    logger.info("Test 4: Sentence Transformer Model Loading")
    logger.info("-" * 60)
    results['sentence_transformer'] = test_sentence_transformer_loading()
    logger.info("")
    
    # Test 5: PDF parser basic
    logger.info("Test 5: PDF Parser Basic")
    logger.info("-" * 60)
    results['pdf_parser'] = test_pdf_parser_basic()
    logger.info("")
    
    # Test 6: Layout analyzer basic
    logger.info("Test 6: Layout Analyzer Basic")
    logger.info("-" * 60)
    results['layout_analyzer'] = test_layout_analyzer_basic()
    logger.info("")
    
    # Test 7: Header/footer detector basic
    logger.info("Test 7: Header/Footer Detector Basic")
    logger.info("-" * 60)
    results['header_footer'] = test_header_footer_detector_basic()
    logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info("%s: %s", test_name.replace('_', ' ').title(), status)
    
    logger.info("")
    if all_passed:
        logger.info("✓ All extraction module tests PASSED!")
        logger.info("")
        logger.info("Models are properly configured and can be loaded:")
        logger.info("  - DeepSeek-OCR: Ready for OCR processing")
        logger.info("  - Sentence Transformer: Ready for text comparison")
        sys.exit(0)
    else:
        logger.error("✗ Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

