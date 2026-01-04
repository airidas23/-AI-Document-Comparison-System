"""Text normalization utilities for comparison."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional


# Unicode characters to normalize
_SPECIAL_SPACES = [
    "\u00A0",  # NBSP (Non-Breaking Space)
    "\u202F",  # Narrow NBSP
    "\u2007",  # Figure Space
    "\u2009",  # Thin Space
    "\u200A",  # Hair Space
    "\u3000",  # Ideographic Space
]

_ZERO_WIDTH_CHARS = [
    "\u200B",  # Zero-Width Space
    "\u200C",  # Zero-Width Non-Joiner
    "\u200D",  # Zero-Width Joiner
    "\uFEFF",  # BOM / Zero-Width No-Break Space
    "\u00AD",  # Soft Hyphen
    "\u2060",  # Word Joiner
]

_DASH_VARIANTS = [
    "\u2010",  # Hyphen
    "\u2011",  # Non-Breaking Hyphen
    "\u2012",  # Figure Dash
    "\u2013",  # En Dash
    "\u2014",  # Em Dash
    "\u2015",  # Horizontal Bar
    "\u2212",  # Minus Sign
]


def normalize_text_full(
    text: str,
    *,
    preserve_case: bool = False,
    normalize_dashes: bool = True,
    strip_zero_width: bool = True,
    normalize_spaces: bool = True,
) -> str:
    """
    Full text normalization for academic PDF comparison.
    
    This function performs comprehensive normalization to ensure accurate
    text comparison across different PDF extraction methods and encodings.
    Critical for Lithuanian and other Unicode-heavy languages.
    
    Normalization steps:
    1. NFC Unicode normalization (composite Lithuanian chars like ą, č, ę, ė, į, š, ų, ū, ž)
    2. Special spaces (NBSP, thin space, etc.) → regular space
    3. Zero-width characters removal (ZWSP, ZWNJ, soft hyphen, etc.)
    4. Dash variant normalization (en-dash, em-dash → hyphen)
    5. Whitespace collapse (multiple spaces → single space)
    6. Optional lowercase conversion
    
    Args:
        text: Input text to normalize
        preserve_case: If True, don't convert to lowercase (default: False)
        normalize_dashes: If True, convert dash variants to ASCII hyphen (default: True)
        strip_zero_width: If True, remove zero-width characters (default: True)
        normalize_spaces: If True, convert special spaces to regular space (default: True)
    
    Returns:
        Normalized text string suitable for comparison
    
    Examples:
        >>> normalize_text_full("Ąžuolas ėjo į mokyklą")
        'ąžuolas ėjo į mokyklą'
        
        >>> normalize_text_full("Hello\\u00A0World")  # NBSP
        'hello world'
        
        >>> normalize_text_full("Test\\u200BWord")  # Zero-width space
        'testword'
        
        >>> normalize_text_full("2020\\u20132021")  # En-dash
        '2020-2021'
        
        >>> normalize_text_full("Lietuva", preserve_case=True)
        'Lietuva'
    """
    if not text:
        return ""
    
    # Step 1: NFC normalization - critical for Lithuanian characters
    # This ensures composed forms: 'a' + combining cedilla → 'ą'
    result = unicodedata.normalize("NFC", text)
    
    # Step 2: Special spaces → regular space
    if normalize_spaces:
        for char in _SPECIAL_SPACES:
            result = result.replace(char, " ")
    
    # Step 3: Remove zero-width characters
    if strip_zero_width:
        for char in _ZERO_WIDTH_CHARS:
            result = result.replace(char, "")
    
    # Step 4: Normalize dash variants to ASCII hyphen
    if normalize_dashes:
        for char in _DASH_VARIANTS:
            result = result.replace(char, "-")
        # Also normalize spacing around dashes
        result = re.sub(r"\s*-\s*", "-", result)
    
    # Step 5: Collapse whitespace (multiple spaces/tabs/newlines → single space)
    result = re.sub(r"\s+", " ", result)
    
    # Step 6: Strip leading/trailing whitespace
    result = result.strip()
    
    # Step 7: Optional lowercase
    if not preserve_case:
        result = result.lower()
    
    return result


def normalize_text(text: str, ocr: bool = False) -> str:
    """
    Normalize text for comparison by ignoring case, whitespace, and minor character differences.
    
    This function:
    - Converts text to lowercase
    - Normalizes Unicode characters (NFD/NFC) for proper character comparison
    - Normalizes whitespace (collapses multiple spaces)
    - Strips leading/trailing whitespace
    
    Args:
        text: Input text to normalize
        ocr: If True, apply OCR-specific normalization (dash handling, hyphenation)
    
    Returns:
        Normalized text string
    
    Examples:
        >>> normalize_text("John")
        'john'
        >>> normalize_text("Žmogaus 3D modelio konstravimas")
        'žmogaus 3d modelio konstravimas'
        >>> normalize_text("  Multiple   Spaces  ")
        'multiple spaces'
    """
    if not text:
        return ""
    
    # Convert to lowercase
    normalized = text.lower()
    
    # Normalize Unicode (NFD -> NFC) to handle accented characters consistently
    # This ensures "Ž" and similar characters are compared correctly
    normalized = unicodedata.normalize("NFC", normalized)

    if ocr:
        # OCR often confuses dash variants and spacing around them.
        # Map common Unicode dashes/minus to ASCII hyphen and normalize surrounding whitespace.
        normalized = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", normalized)
        normalized = re.sub(r"\s*-\s*", "-", normalized)
        # Handle hyphenation: "foo-\nbar" -> "foobar"
        normalized = re.sub(r"-\s*\n\s*", "", normalized)
    
    # Normalize whitespace: collapse multiple spaces/tabs/newlines to single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized


def normalize_text_for_matching(text: str, ocr: bool = False) -> str:
    """
    Normalize text for OCR matching with focus on reducing false positives.
    
    This preserves punctuation and character content while normalizing:
    - Whitespace (spaces, tabs, newlines -> single space)
    - Hyphenation (word-\nbreak -> wordbreak)
    - Unicode to NFC form
    - Case to lowercase
    
    IMPORTANT: Does NOT remove punctuation - commas, periods etc. are preserved
    so that punctuation changes are detected as real diffs.
    
    Args:
        text: Input text to normalize
        ocr: If True, apply OCR-specific hyphenation handling
    
    Returns:
        Normalized text for matching
    """
    if not text:
        return ""
    
    # NFC normalization for consistent Unicode handling
    result = unicodedata.normalize("NFC", text)
    
    # Lowercase for case-insensitive matching
    result = result.lower()
    
    if ocr:
        # Handle hyphenation: "foo-\nbar" or "foo- bar" -> "foobar"
        # This is OCR-specific where line breaks are often incorrect
        result = re.sub(r"-\s*[\n\r]+\s*", "", result)
        result = re.sub(r"-\s+", "-", result)  # normalize spacing around remaining hyphens
        
        # Normalize dash variants to ASCII hyphen
        result = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", result)
    
    # Collapse all whitespace to single space
    result = re.sub(r"\s+", " ", result)
    
    # Strip leading/trailing whitespace
    result = result.strip()
    
    return result


def compute_ocr_change_significance(text_a: str, text_b: str, ocr: bool = True) -> dict:
    """
    Compute significance metrics for an OCR text change.
    
    Returns dict with:
    - changed_chars: number of character edits (Levenshtein distance)
    - max_len: max(len(text_a), len(text_b))
    - change_ratio: changed_chars / max_len
    - is_significant: True if change meets OCR significance thresholds
    
    Args:
        text_a: Original text
        text_b: Modified text
        ocr: Whether to use OCR-specific normalization
    
    Returns:
        Dictionary with change significance metrics
    """
    from rapidfuzz.distance import Indel
    
    # Normalize for comparison (but preserve punctuation/content)
    norm_a = normalize_text_for_matching(text_a, ocr=ocr)
    norm_b = normalize_text_for_matching(text_b, ocr=ocr)
    
    # If normalized texts are identical, it's just whitespace/formatting noise
    if norm_a == norm_b:
        return {
            "changed_chars": 0,
            "max_len": max(len(text_a), len(text_b)),
            "change_ratio": 0.0,
            "is_significant": False,
            "is_whitespace_only": text_a != text_b,
        }
    
    # Calculate character-level edit distance using rapidfuzz (89x faster than difflib)
    # Indel.opcodes returns Opcode namedtuples with (tag, src_start, src_end, dest_start, dest_end)
    changed_chars = 0
    for op in Indel.opcodes(norm_a, norm_b):
        if op.tag != "equal":
            changed_chars += max(op.src_end - op.src_start, op.dest_end - op.dest_start)
    
    max_len = max(len(norm_a), len(norm_b), 1)
    change_ratio = changed_chars / max_len
    
    # Import settings for thresholds
    try:
        from config.settings import settings
        min_chars = settings.ocr_min_change_chars
        min_ratio = settings.ocr_min_change_ratio
    except ImportError:
        min_chars = 2
        min_ratio = 0.015
    
    is_significant = changed_chars >= min_chars and change_ratio >= min_ratio
    
    return {
        "changed_chars": changed_chars,
        "max_len": max_len,
        "change_ratio": change_ratio,
        "is_significant": is_significant,
        "is_whitespace_only": False,
    }

