"""Test that the app can start and initialize correctly."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging import configure_logging, logger

configure_logging()


def test_app_initialization() -> bool:
    """Test that the app can be initialized."""
    logger.info("=" * 60)
    logger.info("App Startup Test")
    logger.info("=" * 60)
    logger.info("")
    
    try:
        # Test 1: Import app module
        logger.info("Test 1: Import App Module")
        logger.info("-" * 60)
        from app import main
        logger.info("  ✓ app.py imported successfully")
        logger.info("")
        
        # Test 2: Import Gradio UI
        logger.info("Test 2: Import Gradio UI")
        logger.info("-" * 60)
        from visualization.gradio_ui import build_comparison_interface
        logger.info("  ✓ gradio_ui imported successfully")
        logger.info("")
        
        # Test 3: Build Interface (without launching)
        logger.info("Test 3: Build Gradio Interface")
        logger.info("-" * 60)
        interface = build_comparison_interface()
        logger.info("  ✓ Gradio interface built successfully")
        logger.info("  ✓ Interface type: %s", type(interface).__name__)
        logger.info("")
        
        # Test 4: Verify Settings
        logger.info("Test 4: Verify Settings")
        logger.info("-" * 60)
        from config.settings import settings
        logger.info("  ✓ Settings loaded")
        logger.info("  - DeepSeek-OCR: %s", settings.deepseek_ocr_model_path)
        logger.info("  - Sentence Transformer: %s", settings.sentence_transformer_model)
        logger.info("  - Max pages: %d", settings.max_pages)
        logger.info("  - Performance target: <%.1fs/page", settings.seconds_per_page_target)
        logger.info("")
        
        # Test 5: Verify All Dependencies
        logger.info("Test 5: Verify Dependencies")
        logger.info("-" * 60)
        try:
            import gradio
            logger.info("  ✓ Gradio available (version: %s)", gradio.__version__)
        except ImportError:
            logger.error("  ✗ Gradio not available")
            return False
        
        try:
            import fitz  # PyMuPDF
            logger.info("  ✓ PyMuPDF available")
        except ImportError:
            logger.error("  ✗ PyMuPDF not available")
            return False
        
        try:
            import sentence_transformers
            logger.info("  ✓ sentence-transformers available")
        except ImportError:
            logger.error("  ✗ sentence-transformers not available")
            return False
        
        try:
            from transformers import AutoModel
            logger.info("  ✓ transformers available")
        except ImportError:
            logger.error("  ✗ transformers not available")
            return False
        
        logger.info("")
        
        logger.info("=" * 60)
        logger.info("✓ App Startup Test PASSED")
        logger.info("=" * 60)
        logger.info("")
        logger.info("The application is ready to run!")
        logger.info("To start the app, run: python3 app.py")
        logger.info("Then open http://localhost:7860 in your browser")
        
        return True
        
    except Exception as exc:
        logger.error("✗ App Startup Test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main() -> None:
    """Run the app startup test."""
    success = test_app_initialization()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()



