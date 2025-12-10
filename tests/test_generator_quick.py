#!/usr/bin/env python3
"""Quick test script to verify synthetic PDF generation with 1 variation."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.synthetic.generator import generate_synthetic_pairs
from utils.logging import configure_logging, logger

configure_logging()

def main():
    """Run a quick 1-variation test."""
    # Create test directories
    test_dir = project_root / "data" / "synthetic" / "test_output"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a minimal base PDF for testing
    base_pdf = test_dir / "base_test.pdf"
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        
        doc = SimpleDocTemplate(str(base_pdf), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("Test Document for Synthetic Generation", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("This is a simple test document used to verify the synthetic PDF generator.", styles["BodyText"]))
        story.append(Paragraph("It contains minimal content that will be used as a base for generating variations.", styles["BodyText"]))
        doc.build(story)
        logger.info("Created base PDF: %s", base_pdf)
    except Exception as exc:
        logger.error("Failed to create base PDF: %s", exc)
        return 1
    
    # Run generation with 1 variation
    logger.info("Generating 1 variation...")
    try:
        pairs = generate_synthetic_pairs(
            base_pdf_path=base_pdf,
            output_dir=test_dir,
            num_variations=1,
        )
        
        if pairs:
            logger.info("✓ Generation successful!")
            logger.info("Generated %d pair(s)", len(pairs))
            pair = pairs[0]
            logger.info("Pair ID: %s", pair.get("pair_id"))
            logger.info("Original PDF: %s", pair.get("original_pdf"))
            logger.info("Modified PDF: %s", pair.get("modified_pdf"))
            logger.info("Changes logged: %d", len(pair.get("changes", [])))
            
            # Verify files exist
            original_path = Path(pair.get("original_pdf", ""))
            modified_path = Path(pair.get("modified_pdf", ""))
            
            if original_path.exists() and modified_path.exists():
                logger.info("✓ Both PDF files created successfully")
                logger.info("  Original size: %d bytes", original_path.stat().st_size)
                logger.info("  Modified size: %d bytes", modified_path.stat().st_size)
                return 0
            else:
                logger.error("✗ PDF files not found")
                return 1
        else:
            logger.error("✗ No pairs generated")
            return 1
            
    except Exception as exc:
        logger.error("✗ Generation failed: %s", exc, exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
