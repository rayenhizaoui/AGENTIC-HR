"""
CV Parsing Tools

This module contains tools for parsing CVs/resumes from various formats,
cleaning text, and handling batch uploads.
"""

import re
import io
import unicodedata
from typing import Optional, Any, List
from langchain_core.tools import tool

try:
    import docx
except ImportError:
    docx = None
    print("Warning: python-docx not installed. DOCX parsing will fail.")

# PDF extraction: try multiple libraries for maximum compatibility
_pdf_extractors = []

try:
    import pdfplumber
    _pdf_extractors.append("pdfplumber")
except ImportError:
    pdfplumber = None

try:
    import fitz  # pymupdf
    _pdf_extractors.append("pymupdf")
except ImportError:
    fitz = None

try:
    import PyPDF2
    _pdf_extractors.append("PyPDF2")
except ImportError:
    PyPDF2 = None

if not _pdf_extractors:
    print("Warning: No PDF library installed (pdfplumber, pymupdf, PyPDF2). PDF parsing will fail.")
else:
    print(f"PDF extractors available: {', '.join(_pdf_extractors)}")


# ── Text Cleaning Pipeline ──────────────────────────────────

@tool
def text_cleaner_pipeline(text: str) -> dict:
    """
    Clean and normalize raw text extracted from documents.

    Strips whitespace, removes emojis, normalizes unicode,
    and cleans up formatting artifacts.

    Args:
        text: Raw text to clean.

    Returns:
        Dictionary with 'cleaned_text' and 'stats'.
    """
    if not text:
        return {"cleaned_text": "", "stats": {"original_length": 0, "cleaned_length": 0}}

    original_length = len(text)

    # 1. Normalize unicode (NFKD → recompose to NFC)
    cleaned = unicodedata.normalize("NFKC", text)

    # 2. Remove emojis and special unicode symbols
    cleaned = re.sub(
        r"[^\w\s\.\,\-\+\#\/\(\)\[\]\:\;\@\&\%\!\?\'\"\=\>\<]",
        " ",
        cleaned,
        flags=re.UNICODE
    )

    # 3. Normalize whitespace (multiple spaces → single)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # 4. Normalize line breaks (multiple blank lines → double newline)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)

    # 5. Strip each line
    lines = [line.strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)

    # 6. Final strip
    cleaned = cleaned.strip()

    return {
        "cleaned_text": cleaned,
        "stats": {
            "original_length": original_length,
            "cleaned_length": len(cleaned),
            "reduction_percent": round((1 - len(cleaned) / max(original_length, 1)) * 100, 1)
        }
    }


def clean_text(text):
    """Basic text cleaning function (legacy helper)."""
    if not text:
        return ""
    result = text_cleaner_pipeline.invoke({"text": text})
    return result["cleaned_text"]


# ── CV Parser Tool ──────────────────────────────────────────

@tool
def cv_parser_tool(file_obj: Any) -> dict:
    """
    Parse a CV/Resume file object (Streamlit UploadedFile) and extract its text content.

    Supports PDF and DOCX formats.

    Args:
        file_obj: The file object (BytesIO-like) uploaded by the user.
                  Must have .name attribute ending in .pdf or .docx.

    Returns:
        A dictionary containing:
        - filename: Name of the file
        - filetype: 'pdf' or 'docx'
        - text: Cleaned extracted text
        - pages: Number of pages
        - word_count: Number of words
        - ocr_required: Boolean if OCR might be needed
        - error: Error message if parsing failed
    """
    result = {
        "filename": getattr(file_obj, "name", "unknown"),
        "filetype": None,
        "text": "",
        "pages": 0,
        "word_count": 0,
        "ocr_required": False,
        "error": None
    }

    if not hasattr(file_obj, "read"):
        result["error"] = "Invalid file object provided."
        return result

    filename = result["filename"]

    # PDF
    if filename.lower().endswith('.pdf'):
        result["filetype"] = "pdf"
        text = ""
        extraction_method = None

        # --- Strategy 1: pdfplumber (best for most PDFs) ---
        if pdfplumber and not text.strip():
            try:
                file_obj.seek(0)
                with pdfplumber.open(file_obj) as pdf:
                    result["pages"] = len(pdf.pages)
                    page_texts = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            page_texts.append(page_text)
                    text = "\n".join(page_texts)
                    if text.strip():
                        extraction_method = "pdfplumber"
            except Exception as e:
                print(f"pdfplumber failed: {e}")

        # --- Strategy 2: pymupdf/fitz (handles complex layouts) ---
        if fitz and not text.strip():
            try:
                file_obj.seek(0)
                pdf_bytes = file_obj.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                result["pages"] = len(doc)
                page_texts = []
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        page_texts.append(page_text)
                doc.close()
                text = "\n".join(page_texts)
                if text.strip():
                    extraction_method = "pymupdf"
            except Exception as e:
                print(f"pymupdf failed: {e}")

        # --- Strategy 3: PyPDF2 (legacy fallback) ---
        if PyPDF2 and not text.strip():
            try:
                file_obj.seek(0)
                reader = PyPDF2.PdfReader(file_obj)
                result["pages"] = len(reader.pages)
                page_texts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        page_texts.append(page_text)
                text = "\n".join(page_texts)
                if text.strip():
                    extraction_method = "PyPDF2"
            except Exception as e:
                print(f"PyPDF2 failed: {e}")

        if not text.strip():
            # --- Strategy 4: OCR fallback (scanned/image-based PDFs) ---
            try:
                from .ocr_tool import ocr_cv_tool
                file_obj.seek(0)
                pdf_bytes = file_obj.read()
                ocr_result = ocr_cv_tool.invoke({
                    "file_path": "",
                    "file_bytes": pdf_bytes,
                    "languages": ["en", "fr"],
                })
                ocr_text = ocr_result.get("text", "")
                if ocr_text.strip():
                    text = ocr_text
                    extraction_method = f"ocr ({ocr_result.get('ocr_backend', 'unknown')})"
                    result["pages"] = ocr_result.get("pages", 0)
                    result["ocr_confidence"] = ocr_result.get("confidence", 0.0)
                    print(f"OCR succeeded with {ocr_result.get('ocr_backend')}, "
                          f"confidence={ocr_result.get('confidence', 0):.2f}")
            except Exception as e:
                print(f"OCR fallback failed: {e}")

        if not text.strip():
            result["text"] = ""
            result["word_count"] = 0
            result["ocr_required"] = True
            result["error"] = (
                "Could not extract text from this PDF. "
                "The file may be image-based/scanned and no OCR backend is available. "
                "Install easyocr or pytesseract for scanned PDF support."
            )
            return result

        cleaned_text = clean_text(text)
        result["text"] = cleaned_text
        result["word_count"] = len(cleaned_text.split())
        result["extraction_method"] = extraction_method

        if result["word_count"] < 50:
            result["ocr_required"] = True
            # Try OCR to supplement low-quality text extraction
            try:
                from .ocr_tool import ocr_cv_tool
                file_obj.seek(0)
                pdf_bytes = file_obj.read()
                ocr_result = ocr_cv_tool.invoke({
                    "file_path": "",
                    "file_bytes": pdf_bytes,
                    "languages": ["en", "fr"],
                })
                ocr_text = ocr_result.get("text", "")
                if ocr_text.strip() and len(ocr_text.split()) > result["word_count"]:
                    result["text"] = clean_text(ocr_text)
                    result["word_count"] = len(result["text"].split())
                    result["extraction_method"] = f"ocr ({ocr_result.get('ocr_backend', 'unknown')})"
                    result["ocr_confidence"] = ocr_result.get("confidence", 0.0)
                    result["ocr_required"] = False
            except Exception as e:
                print(f"OCR supplement for low-word-count PDF failed: {e}")

    # DOCX
    elif filename.lower().endswith('.docx'):
        result["filetype"] = "docx"
        try:
            doc = docx.Document(file_obj)

            text = "\n".join(p.text for p in doc.paragraphs if p.text)
            cleaned_text = clean_text(text)

            result["text"] = cleaned_text
            result["word_count"] = len(cleaned_text.split())

            if result["word_count"] < 50:
                result["ocr_required"] = True

        except Exception as e:
            result["error"] = f"DOCX parsing error: {e}"
            return result

    # IMAGE files (JPG, PNG, etc.) — OCR only
    elif any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']):
        result["filetype"] = "image"
        try:
            from .ocr_tool import ocr_cv_tool
            file_obj.seek(0)
            img_bytes = file_obj.read()
            ocr_result = ocr_cv_tool.invoke({
                "file_path": "",
                "file_bytes": img_bytes,
                "languages": ["en", "fr"],
            })
            ocr_text = ocr_result.get("text", "")
            if ocr_text.strip():
                result["text"] = clean_text(ocr_text)
                result["word_count"] = len(result["text"].split())
                result["pages"] = 1
                result["extraction_method"] = f"ocr ({ocr_result.get('ocr_backend', 'unknown')})"
                result["ocr_confidence"] = ocr_result.get("confidence", 0.0)
            else:
                result["error"] = (
                    "OCR could not extract text from this image. "
                    "The image may be too low quality or no OCR backend is installed."
                )
                result["ocr_required"] = True
        except Exception as e:
            result["error"] = f"Image OCR error: {e}"
            result["ocr_required"] = True

    else:
        result["error"] = "Unsupported file type. Supported: PDF, DOCX, JPG, PNG."
        return result

    return result


# ── Batch Upload Handler ────────────────────────────────────

@tool
def batch_cv_parser(file_objects: List[Any]) -> dict:
    """
    Parse multiple CV files at once. The agent can loop through 10+ CVs in a batch.

    Args:
        file_objects: A list of file objects (BytesIO-like), each with a .name attribute.

    Returns:
        Dictionary with:
        - total: Number of files processed
        - successful: Number successfully parsed
        - failed: Number that failed
        - results: List of individual parse results
    """
    results = []
    successful = 0
    failed = 0

    for file_obj in file_objects:
        try:
            parsed = cv_parser_tool.invoke({"file_obj": file_obj})
            results.append(parsed)
            if parsed.get("error"):
                failed += 1
            else:
                successful += 1
        except Exception as e:
            results.append({
                "filename": getattr(file_obj, "name", "unknown"),
                "error": str(e)
            })
            failed += 1

    return {
        "total": len(file_objects),
        "successful": successful,
        "failed": failed,
        "results": results
    }
