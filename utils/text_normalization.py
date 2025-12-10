"""Text normalization utilities for comparison."""
from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by ignoring case, whitespace, and minor character differences.
    
    This function:
    - Converts text to lowercase
    - Normalizes Unicode characters (NFD/NFC) for proper character comparison
    - Normalizes whitespace (collapses multiple spaces)
    - Strips leading/trailing whitespace
    
    Args:
        text: Input text to normalize
    
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
    
    # Normalize whitespace: collapse multiple spaces/tabs/newlines to single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized

