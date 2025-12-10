"""Style normalization utilities for consistent formatting comparison."""
from __future__ import annotations

from typing import Dict


# Font name mapping for normalization
FONT_NAME_MAPPING: Dict[str, str] = {
    # Times family variants
    "timesnewromanpsmt": "times new roman",
    "timesnewroman": "times new roman",
    "timesnewromanps": "times new roman",
    "times-roman": "times new roman",
    "timesnew": "times new roman",
    "times": "times new roman",
    "tnr": "times new roman",
    
    # Arial/Helvetica family
    "arialmt": "arial",
    "arial": "arial",
    "arialunicode": "arial",
    "helvetica": "arial",
    "helveticaneue": "arial",
    "helvetica neue": "arial",
    
    # Courier family
    "couriernew": "courier new",
    "couriernewps": "courier new",
    "courier": "courier new",
    
    # Calibri
    "calibri": "calibri",
    "calibril": "calibri",
    
    # Georgia
    "georgia": "georgia",
    
    # Verdana
    "verdana": "verdana",
    
    # Cambria
    "cambria": "cambria",
    
    # Comic Sans
    "comicsans": "comic sans ms",
    "comicsansms": "comic sans ms",
}


def normalize_font_name(font_name: str) -> str:
    """
    Normalize font names to a standard format.
    
    Examples:
        "TimesNewRomanPSMT" -> "times new roman"
        "ArialMT" -> "arial"
        "Helvetica" -> "arial"
    
    Args:
        font_name: Original font name
    
    Returns:
        Normalized font name (lowercase, standard name)
    """
    if not font_name:
        return ""
    
    # Convert to lowercase and remove common suffixes
    normalized = font_name.lower().strip()
    
    # Remove common PDF font suffixes (in order of specificity)
    suffixes_to_remove = [
        "-bolditalic", "-bold-italic", "bolditalic", "bold-italic",
        "-bold", "bold",
        "-italic", "italic",
        "mt", "ps", "psmt", "std", "regular"
    ]
    
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()
    
    # Remove spaces, hyphens, and numbers for better matching
    normalized_no_spaces = normalized.replace(" ", "").replace("-", "").replace("_", "")
    # Remove trailing numbers (e.g., "arial1" -> "arial")
    normalized_clean = ''.join(c for c in normalized_no_spaces if not c.isdigit())
    
    # Check mapping with both spaced and non-spaced versions
    if normalized in FONT_NAME_MAPPING:
        return FONT_NAME_MAPPING[normalized]
    if normalized_no_spaces in FONT_NAME_MAPPING:
        return FONT_NAME_MAPPING[normalized_no_spaces]
    if normalized_clean in FONT_NAME_MAPPING:
        return FONT_NAME_MAPPING[normalized_clean]
    
    # Try partial matching for known font families (check if mapping key is contained)
    for key, value in FONT_NAME_MAPPING.items():
        if key in normalized or key in normalized_no_spaces or key in normalized_clean:
            return value
        # Also check reverse (normalized contains key)
        if normalized.startswith(key) or normalized_no_spaces.startswith(key):
            return value
    
    # Return normalized version
    return normalized


def normalize_font_size(size: float, bucket_size: float = 0.5) -> float:
    """
    Round font size to a bucket to reduce noise.
    
    Examples:
        normalize_font_size(12.3, 0.5) -> 12.5
        normalize_font_size(12.7, 0.5) -> 12.5
        normalize_font_size(12.3, 0.1) -> 12.3
    
    Args:
        size: Font size in points
        bucket_size: Size of bucket for rounding (default: 0.5pt)
    
    Returns:
        Rounded font size
    """
    if size is None or size <= 0:
        return 0.0
    
    # Round to nearest bucket
    return round(size / bucket_size) * bucket_size


def get_font_family_normalized(font_name: str) -> str:
    """Get normalized font family name."""
    return normalize_font_name(font_name)


def get_size_bucket(size: float, bucket_size: float = 0.5) -> float:
    """Get font size bucket."""
    return normalize_font_size(size, bucket_size)
