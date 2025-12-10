"""Character-level text comparison utilities for detecting text changes."""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Dict, List


def detect_character_changes(text_a: str, text_b: str) -> Dict:
    """
    Detect character-level changes between two texts, excluding whitespace/punctuation differences.
    
    This function compares texts at the character level to identify actual content changes
    (character additions, deletions, modifications) separate from formatting changes
    (whitespace, punctuation).
    
    Args:
        text_a: First text to compare
        text_b: Second text to compare
    
    Returns:
        Dictionary with:
        - has_character_change: bool - True if character content differs
        - character_diff_ratio: float - Ratio of changed characters (0.0-1.0)
        - changed_chars: List[tuple] - List of (position, old_char, new_char) tuples
        - normalized_a: str - Normalized version of text_a (whitespace normalized)
        - normalized_b: str - Normalized version of text_b (whitespace normalized)
    
    Examples:
        >>> result = detect_character_changes("konstravimas", "konstravima")
        >>> result["has_character_change"]
        True
        >>> result = detect_character_changes("Hello  World", "Hello World")
        >>> result["has_character_change"]
        False
    """
    if not text_a and not text_b:
        return {
            "has_character_change": False,
            "character_diff_ratio": 0.0,
            "changed_chars": [],
            "normalized_a": "",
            "normalized_b": "",
        }
    
    # Normalize whitespace for comparison (collapse multiple spaces/newlines to single space)
    # but keep the original character content
    normalized_a = re.sub(r'\s+', ' ', text_a.strip())
    normalized_b = re.sub(r'\s+', ' ', text_b.strip())
    
    # Normalize Unicode for proper character comparison
    normalized_a = unicodedata.normalize("NFC", normalized_a)
    normalized_b = unicodedata.normalize("NFC", normalized_b)
    
    # Use SequenceMatcher to find character-level differences
    matcher = difflib.SequenceMatcher(None, normalized_a, normalized_b)
    
    # Extract character changes (excluding whitespace-only differences)
    changed_chars: List[tuple] = []
    total_chars = max(len(normalized_a), len(normalized_b), 1)
    changed_char_count = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Characters were replaced
            old_chars = normalized_a[i1:i2]
            new_chars = normalized_b[j1:j2]
            
            # Check if this is more than just whitespace/punctuation change
            old_content = re.sub(r'[\s\W]', '', old_chars)
            new_content = re.sub(r'[\s\W]', '', new_chars)
            
            if old_content != new_content:
                # Actual character content changed
                for idx, (old_char, new_char) in enumerate(zip(old_chars, new_chars)):
                    if old_char != new_char and not (_is_whitespace_or_punct(old_char) and _is_whitespace_or_punct(new_char)):
                        changed_chars.append((i1 + idx, old_char, new_char))
                        changed_char_count += 1
                # Handle length differences
                if len(old_chars) != len(new_chars):
                    changed_char_count += abs(len(old_chars) - len(new_chars))
        elif tag == 'delete':
            # Characters were deleted
            deleted_chars = normalized_a[i1:i2]
            deleted_content = re.sub(r'[\s\W]', '', deleted_chars)
            if deleted_content:  # Not just whitespace/punctuation
                changed_chars.append((i1, deleted_chars, ""))
                changed_char_count += len(deleted_content)
        elif tag == 'insert':
            # Characters were inserted
            inserted_chars = normalized_b[j1:j2]
            inserted_content = re.sub(r'[\s\W]', '', inserted_chars)
            if inserted_content:  # Not just whitespace/punctuation
                changed_chars.append((i1, "", inserted_chars))
                changed_char_count += len(inserted_content)
    
    # Calculate difference ratio
    character_diff_ratio = changed_char_count / total_chars if total_chars > 0 else 0.0
    
    # Determine if there's a meaningful character change
    # For short texts (< 50 chars), any character change is significant
    # For longer texts, use a threshold (1% difference or at least 1 character)
    has_character_change = False
    if len(normalized_a) < 50 or len(normalized_b) < 50:
        # Short text: any character change is significant
        has_character_change = character_diff_ratio > 0.0 or len(changed_chars) > 0
    else:
        # Longer text: use threshold (1% or at least 1 character)
        has_character_change = character_diff_ratio > 0.01 or len(changed_chars) > 0
    
    return {
        "has_character_change": has_character_change,
        "character_diff_ratio": character_diff_ratio,
        "changed_chars": changed_chars,
        "normalized_a": normalized_a,
        "normalized_b": normalized_b,
    }


def _is_whitespace_or_punct(char: str) -> bool:
    """Check if character is whitespace or punctuation."""
    return char.isspace() or (not char.isalnum() and not unicodedata.category(char).startswith('L'))


def has_character_content_difference(text_a: str, text_b: str) -> bool:
    """
    Quick check if texts differ in character content (excluding whitespace/punctuation).
    
    This is a faster version that just returns True/False without detailed diff information.
    
    Args:
        text_a: First text
        text_b: Second text
    
    Returns:
        True if character content differs, False otherwise
    """
    # Normalize whitespace
    normalized_a = re.sub(r'\s+', ' ', text_a.strip())
    normalized_b = re.sub(r'\s+', ' ', text_b.strip())
    
    # Normalize Unicode
    normalized_a = unicodedata.normalize("NFC", normalized_a)
    normalized_b = unicodedata.normalize("NFC", normalized_b)
    
    # Remove all whitespace and punctuation for content comparison
    content_a = re.sub(r'[\s\W]', '', normalized_a)
    content_b = re.sub(r'[\s\W]', '', normalized_b)
    
    return content_a != content_b
