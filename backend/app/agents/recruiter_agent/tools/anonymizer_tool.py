"""
Enhanced PII Anonymizer Tool

Removes Personally Identifiable Information from CV text to reduce hiring bias.
Covers: emails, phones, names, addresses, dates of birth, social media URLs,
national IDs (Tunisian CIN, SSN), and photo references.
"""

import re
from langchain_core.tools import tool

# ── PII Detection Patterns ────────────────────────────────────

EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

PHONE_PATTERN = re.compile(
    r"(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}"
)

NAME_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})\b"
)

# Tunisian CIN (8 digits) and generic SSN-like patterns
NATIONAL_ID_PATTERN = re.compile(
    r"\b\d{8}\b|\b\d{3}-\d{2}-\d{4}\b"
)

# Date of birth patterns (DD/MM/YYYY, YYYY-MM-DD, etc.)
DOB_PATTERN = re.compile(
    r"(?:date\s*(?:de\s*)?(?:naissance|of\s*birth|birth)\s*[:\-]?\s*)"
    r"(\d{1,2}[\s/\-\.]\d{1,2}[\s/\-\.]\d{2,4}|\d{4}[\s/\-\.]\d{1,2}[\s/\-\.]\d{1,2})",
    re.IGNORECASE,
)

# Standalone birth dates (common formats, but only near keywords to avoid false positives)
STANDALONE_DOB_PATTERN = re.compile(
    r"(?:né(?:e)?\s*(?:le)?\s*|born\s*(?:on)?\s*)"
    r"(\d{1,2}[\s/\-\.]\d{1,2}[\s/\-\.]\d{2,4})",
    re.IGNORECASE,
)

# Physical address patterns (street numbers + common address words)
ADDRESS_PATTERN = re.compile(
    r"\d{1,5}\s*,?\s*(?:rue|avenue|boulevard|street|road|ave|blvd|st|rd|av\.?)\s+[A-Za-zÀ-ÿ\s,]+\d{4,5}",
    re.IGNORECASE,
)

# Social media / personal URLs
SOCIAL_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:linkedin\.com|facebook\.com|twitter\.com|x\.com|instagram\.com)/[^\s]+",
    re.IGNORECASE,
)

# Photo references in documents
PHOTO_PATTERN = re.compile(
    r"(?:photo|image|picture|portrait)\s*[:\-]?\s*\S+\.(?:jpg|jpeg|png|gif|bmp|webp)",
    re.IGNORECASE,
)

# Age mentions
AGE_PATTERN = re.compile(
    r"(?:age|âge)\s*[:\-]?\s*\d{1,2}\s*(?:ans|years?)?",
    re.IGNORECASE,
)

# Gender mentions
GENDER_PATTERN = re.compile(
    r"(?:genre|gender|sexe|sex)\s*[:\-]?\s*(?:male|female|homme|femme|masculin|féminin|M|F)\b",
    re.IGNORECASE,
)

# Marital status
MARITAL_PATTERN = re.compile(
    r"(?:situation\s*(?:familiale|matrimoniale)|marital\s*status|état\s*civil)\s*[:\-]?\s*\w+",
    re.IGNORECASE,
)


@tool
def anonymizer_tool(cv_text: str) -> dict:
    """
    Anonymize CV text by removing all Personally Identifiable Information (PII).
    
    This helps reduce hiring bias by removing names, emails, phone numbers,
    addresses, dates of birth, national IDs, social media links, age, gender,
    marital status, and photo references before processing.

    Args:
        cv_text (str): The text content of the CV.

    Returns:
        dict: {
            "anonymized_text": str,
            "redactions": dict  — count of each PII type removed
        }
    """

    if not cv_text or not cv_text.strip():
        return {"anonymized_text": "", "redactions": {}}

    anonymized = cv_text
    redactions = {}

    # Apply each pattern and count redactions
    patterns = [
        (EMAIL_PATTERN, "[REDACTED_EMAIL]", "emails"),
        (PHONE_PATTERN, "[REDACTED_PHONE]", "phones"),
        (NATIONAL_ID_PATTERN, "[REDACTED_ID]", "national_ids"),
        (DOB_PATTERN, "[REDACTED_DOB]", "dates_of_birth"),
        (STANDALONE_DOB_PATTERN, "[REDACTED_DOB]", "dates_of_birth"),
        (ADDRESS_PATTERN, "[REDACTED_ADDRESS]", "addresses"),
        (SOCIAL_URL_PATTERN, "[REDACTED_URL]", "social_urls"),
        (PHOTO_PATTERN, "[REDACTED_PHOTO]", "photos"),
        (AGE_PATTERN, "[REDACTED_AGE]", "age"),
        (GENDER_PATTERN, "[REDACTED_GENDER]", "gender"),
        (MARITAL_PATTERN, "[REDACTED_MARITAL]", "marital_status"),
        (NAME_PATTERN, "[REDACTED_NAME]", "names"),  # Last to avoid over-matching
    ]

    for pattern, replacement, category in patterns:
        matches = pattern.findall(anonymized)
        if matches:
            redactions[category] = redactions.get(category, 0) + len(matches)
            anonymized = pattern.sub(replacement, anonymized)

    return {
        "anonymized_text": anonymized,
        "redactions": redactions,
    }
