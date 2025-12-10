#!/usr/bin/env python3
"""Generate multiple synthetic PDF variations and analyze results."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.synthetic.generator import generate_synthetic_pairs
from utils.logging import configure_logging, logger

configure_logging()


def create_base_pdf(output_path: Path) -> bool:
    """Create a comprehensive base PDF for testing."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet
        
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("Synthetic PDF Dataset Generation", styles["Title"]))
        story.append(Spacer(1, 12))
        
        # Introduction
        story.append(Paragraph(
            "This document serves as a base template for generating synthetic PDF pairs with known differences. "
            "The generator applies various modifications including text changes, formatting adjustments, and structural alterations.",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 12))
        
        # Section 1
        story.append(Paragraph("Text Modifications", styles["Heading2"]))
        story.append(Paragraph(
            "The system can apply synonym swaps, paraphrasing, typo insertion, and punctuation changes. "
            "These modifications test the ability to detect subtle content differences while maintaining document structure.",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 12))
        
        # Section 2
        story.append(Paragraph("Formatting Changes", styles["Heading2"]))
        story.append(Paragraph(
            "Font size adjustments and spacing modifications are applied to test formatting-aware comparison. "
            "These changes simulate real-world scenarios where documents are reformatted without altering content.",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 12))
        
        # Section 3
        story.append(Paragraph("Structural Modifications", styles["Heading2"]))
        story.append(Paragraph(
            "Table structure changes, figure caption updates, and header/footer modifications provide comprehensive "
            "test cases for layout-aware diff detection. Each variation includes ground truth annotations.",
            styles["BodyText"]
        ))
        story.append(PageBreak())
        
        # Additional content for multi-page documents
        story.append(Paragraph("Extended Content Section", styles["Heading1"]))
        story.append(Paragraph(
            "This page demonstrates how the generator handles multi-page documents. "
            "Paragraphs are distributed across pages, and modifications can occur on any page.",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            "The evaluation framework uses these synthetic pairs to measure detection accuracy, "
            "false positive rates, and the ability to correctly classify different types of changes.",
            styles["BodyText"]
        ))
        
        doc.build(story)
        logger.info("Created comprehensive base PDF: %s", output_path)
        return True
    except Exception as exc:
        logger.error("Failed to create base PDF: %s", exc)
        return False


def analyze_results(pairs: list[dict], output_dir: Path) -> dict:
    """Analyze generated pairs and create statistics."""
    stats = {
        "total_pairs": len(pairs),
        "change_types": Counter(),
        "diff_types": Counter(),
        "severity_distribution": Counter(),
        "modifications_applied": Counter(),
        "changes_per_pair": [],
        "page_distribution": Counter(),
    }
    
    for pair in pairs:
        changes = pair.get("changes", [])
        stats["changes_per_pair"].append(len(changes))
        
        for change in changes:
            stats["change_types"][change.get("change_type", "unknown")] += 1
            stats["diff_types"][change.get("diff_type", "unknown")] += 1
            stats["severity_distribution"][change.get("severity", "unknown")] += 1
            stats["page_distribution"][change.get("page", 0)] += 1
        
        for mod in pair.get("applied_modifications", []):
            stats["modifications_applied"][mod] += 1
    
    # Calculate averages
    if stats["changes_per_pair"]:
        stats["avg_changes_per_pair"] = sum(stats["changes_per_pair"]) / len(stats["changes_per_pair"])
        stats["min_changes"] = min(stats["changes_per_pair"])
        stats["max_changes"] = max(stats["changes_per_pair"])
    
    return stats


def generate_summary_report(stats: dict, pairs: list[dict], output_dir: Path) -> None:
    """Generate a human-readable summary report."""
    report_path = output_dir / "generation_summary.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Synthetic PDF Dataset Generation Summary\n\n")
        f.write(f"**Total Pairs Generated:** {stats['total_pairs']}\n\n")
        
        f.write("## Statistics\n\n")
        f.write(f"- Average changes per pair: {stats.get('avg_changes_per_pair', 0):.2f}\n")
        f.write(f"- Minimum changes in a pair: {stats.get('min_changes', 0)}\n")
        f.write(f"- Maximum changes in a pair: {stats.get('max_changes', 0)}\n\n")
        
        f.write("## Change Type Distribution\n\n")
        for change_type, count in stats["change_types"].most_common():
            f.write(f"- **{change_type}**: {count} occurrences\n")
        f.write("\n")
        
        f.write("## Diff Type Distribution\n\n")
        for diff_type, count in stats["diff_types"].most_common():
            f.write(f"- **{diff_type}**: {count} occurrences\n")
        f.write("\n")
        
        f.write("## Severity Distribution\n\n")
        for severity, count in stats["severity_distribution"].most_common():
            f.write(f"- **{severity}**: {count} occurrences\n")
        f.write("\n")
        
        f.write("## Modification Functions Applied\n\n")
        for mod, count in stats["modifications_applied"].most_common():
            mod_name = mod.replace("_apply_", "").replace("_", " ").title()
            f.write(f"- **{mod_name}**: {count} times\n")
        f.write("\n")
        
        f.write("## Page Distribution\n\n")
        for page, count in sorted(stats["page_distribution"].items()):
            f.write(f"- Page {page}: {count} changes\n")
        f.write("\n")
        
        f.write("## Generated Pairs\n\n")
        for pair in pairs:
            pair_id = pair.get("pair_id", "unknown")
            changes_count = len(pair.get("changes", []))
            modifications = ", ".join(
                m.replace("_apply_", "").replace("_", " ").title() 
                for m in pair.get("applied_modifications", [])
            )
            f.write(f"### {pair_id}\n\n")
            f.write(f"- **Changes:** {changes_count}\n")
            f.write(f"- **Modifications:** {modifications}\n")
            f.write(f"- **Original PDF:** `{Path(pair.get('original_pdf', '')).name}`\n")
            f.write(f"- **Modified PDF:** `{Path(pair.get('modified_pdf', '')).name}`\n")
            f.write(f"- **Change Log:** `{pair_id}_change_log.json`\n\n")
    
    logger.info("Summary report written to: %s", report_path)


def main():
    """Generate multiple variations and analyze results."""
    num_variations = 10
    
    # Setup directories
    output_dir = project_root / "data" / "synthetic" / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create base PDF
    base_pdf = output_dir / "base_document.pdf"
    if not base_pdf.exists():
        logger.info("Creating base PDF...")
        if not create_base_pdf(base_pdf):
            logger.error("Failed to create base PDF")
            return 1
    else:
        logger.info("Using existing base PDF: %s", base_pdf)
    
    # Generate variations
    logger.info("Generating %d synthetic PDF variations...", num_variations)
    try:
        pairs = generate_synthetic_pairs(
            base_pdf_path=base_pdf,
            output_dir=output_dir,
            num_variations=num_variations,
        )
        
        logger.info("✓ Successfully generated %d pairs", len(pairs))
        
        # Analyze results
        logger.info("Analyzing results...")
        stats = analyze_results(pairs, output_dir)
        
        # Save statistics
        stats_path = output_dir / "generation_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info("Statistics saved to: %s", stats_path)
        
        # Generate summary report
        generate_summary_report(stats, pairs, output_dir)
        
        # Print summary to console
        logger.info("\n" + "="*60)
        logger.info("GENERATION SUMMARY")
        logger.info("="*60)
        logger.info("Total pairs: %d", stats["total_pairs"])
        logger.info("Average changes per pair: %.2f", stats.get("avg_changes_per_pair", 0))
        logger.info("Total changes: %d", sum(stats["changes_per_pair"]))
        logger.info("\nChange types:")
        for change_type, count in stats["change_types"].most_common():
            logger.info("  - %s: %d", change_type, count)
        logger.info("\nMost common modifications:")
        for mod, count in stats["modifications_applied"].most_common(5):
            mod_name = mod.replace("_apply_", "").replace("_", " ").title()
            logger.info("  - %s: %d times", mod_name, count)
        logger.info("="*60)
        
        return 0
        
    except Exception as exc:
        logger.error("✗ Generation failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
