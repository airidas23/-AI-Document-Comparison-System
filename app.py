"""Entry point for the AI Document Comparison Gradio app."""
from __future__ import annotations

import os
# Disable tokenizers parallelism to avoid fork warnings when using multiprocessing (e.g., Tesseract OCR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config.settings import settings
from utils.logging import configure_logging, logger
from visualization.gradio_ui import build_comparison_interface


configure_logging()


def main() -> None:
    """Launch the Gradio application."""
    logger.info("Starting AI Document Comparison System")
    try:
        logger.info("Settings: model=%s, threshold=%.2f", 
                    settings.sentence_transformer_model, 
                    settings.text_similarity_threshold)
        logger.info("OCR engine: %s", settings.ocr_engine)
    except Exception:
        # Settings might not be fully initialized yet
        pass
    
    # Warmup OCR engines in background to avoid slow first request
    try:
        from extraction.ocr_router import warmup_ocr_engines
        warmup_ocr_engines(background=True)
        logger.info("OCR warmup started in background")
    except Exception as e:
        logger.warning("OCR warmup failed: %s", e)
    
    interface = build_comparison_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
