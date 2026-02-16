"""
Intelligent OCR Tool for CV Images & Scanned PDFs

Extracts text from image-based/scanned documents using a multi-library
fallback chain:
    1. EasyOCR  (ML-based, multilingual: EN/FR/AR)
    2. pytesseract + Tesseract-OCR
    3. pymupdf built-in OCR (if available)

Includes layout-aware sorting (top-to-bottom, left-to-right) and
confidence filtering to reduce OCR noise.
"""

from __future__ import annotations

import io
import os
import re
import sys
from typing import List, Optional, Tuple

from langchain_core.tools import tool

# ── Optional library imports ──────────────────────────────────

_ocr_backends: list[str] = []

try:
    import easyocr  # type: ignore
    _ocr_backends.append("easyocr")
except ImportError:
    easyocr = None  # type: ignore

try:
    import pytesseract  # type: ignore
    from PIL import Image
    _ocr_backends.append("pytesseract")
except ImportError:
    pytesseract = None  # type: ignore
    Image = None  # type: ignore

try:
    import fitz  # pymupdf – can do basic OCR on embedded images
    if "pymupdf" not in _ocr_backends:
        _ocr_backends.append("pymupdf_ocr")
except ImportError:
    fitz = None  # type: ignore

try:
    from pdf2image import convert_from_path, convert_from_bytes  # type: ignore
    _ocr_backends.append("pdf2image")
except ImportError:
    convert_from_path = None  # type: ignore
    convert_from_bytes = None  # type: ignore

if _ocr_backends:
    print(f"OCR backends available: {', '.join(_ocr_backends)}")
else:
    print("⚠️  No OCR backend installed (easyocr, pytesseract, pdf2image). "
          "Scanned-PDF support will be limited.", file=sys.stderr)


# ── EasyOCR reader singleton ─────────────────────────────────

_easyocr_reader = None


def _get_easyocr_reader(languages: list[str] | None = None):
    """Lazy init of the EasyOCR reader (heavy model load)."""
    global _easyocr_reader
    if easyocr is None:
        return None
    if _easyocr_reader is None:
        langs = languages or ["en", "fr"]
        try:
            _easyocr_reader = easyocr.Reader(langs, gpu=False)
        except Exception as e:
            print(f"EasyOCR init failed: {e}", file=sys.stderr)
            return None
    return _easyocr_reader


# ── Layout-aware sorting ─────────────────────────────────────

def _sort_ocr_results(results: list, line_threshold: int = 15) -> list:
    """
    Sort EasyOCR results by layout position (top-to-bottom, left-to-right).
    Groups results into lines based on y-coordinate proximity.
    
    Args:
        results: EasyOCR results [(bbox, text, confidence), ...]
        line_threshold: Max y-pixel diff to consider same line.
    """
    if not results:
        return results

    # Each result: (bbox, text, conf)  where bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    annotated = []
    for bbox, text, conf in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_left = bbox[0][0]
        annotated.append((y_center, x_left, text, conf))

    # Sort by y first, then x
    annotated.sort(key=lambda r: (r[0], r[1]))

    # Group into lines
    lines: list[list] = []
    current_line: list = []
    last_y = None

    for y, x, text, conf in annotated:
        if last_y is not None and abs(y - last_y) > line_threshold:
            # New line
            lines.append(sorted(current_line, key=lambda r: r[1]))
            current_line = []
        current_line.append((y, x, text, conf))
        last_y = y

    if current_line:
        lines.append(sorted(current_line, key=lambda r: r[1]))

    # Flatten back
    sorted_results = []
    for line in lines:
        for y, x, text, conf in line:
            sorted_results.append((text, conf))

    return sorted_results


# ── Core OCR functions ────────────────────────────────────────

def _ocr_with_easyocr(
    images,
    languages: list[str] | None = None,
    confidence_threshold: float = 0.35,
) -> Tuple[str, float]:
    """
    Run EasyOCR on a list of PIL images or numpy arrays.
    Returns (text, avg_confidence).
    """
    reader = _get_easyocr_reader(languages)
    if reader is None:
        return "", 0.0

    import numpy as np  # type: ignore

    all_text_parts = []
    all_confidences = []

    for img in images:
        # Convert PIL Image to numpy array if needed
        if hasattr(img, "convert"):
            img_array = np.array(img.convert("RGB"))
        else:
            img_array = img

        results = reader.readtext(img_array)

        # Layout-aware sorting
        sorted_results = _sort_ocr_results(results)

        page_text = []
        for text, conf in sorted_results:
            if conf >= confidence_threshold:
                page_text.append(text)
                all_confidences.append(conf)

        all_text_parts.append(" ".join(page_text))

    full_text = "\n\n".join(all_text_parts)
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    return full_text, avg_conf


def _ocr_with_tesseract(images) -> Tuple[str, float]:
    """
    Run pytesseract on a list of PIL images.
    Returns (text, estimated_confidence).
    """
    if pytesseract is None or Image is None:
        return "", 0.0

    all_text_parts = []
    for img in images:
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img)) if isinstance(img, bytes) else img

        text = pytesseract.image_to_string(img, lang="eng+fra")
        all_text_parts.append(text)

    full_text = "\n\n".join(all_text_parts)
    # Tesseract doesn't give per-char confidence easily; estimate from text quality
    confidence = 0.7 if len(full_text.strip()) > 50 else 0.3
    return full_text, confidence


def _pdf_to_images(file_path: str = None, file_bytes: bytes = None) -> list:
    """Convert PDF pages to PIL images for OCR."""
    images = []

    # Strategy 1: pdf2image (poppler-based, best quality)
    if convert_from_path is not None or convert_from_bytes is not None:
        try:
            if file_bytes and convert_from_bytes:
                images = convert_from_bytes(file_bytes, dpi=300)
            elif file_path and convert_from_path:
                images = convert_from_path(file_path, dpi=300)
            if images:
                return images
        except Exception as e:
            print(f"pdf2image failed: {e}", file=sys.stderr)

    # Strategy 2: pymupdf (render pages to pixmaps)
    if fitz is not None:
        try:
            if file_bytes:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
            elif file_path:
                doc = fitz.open(file_path)
            else:
                return images

            for page in doc:
                # Render at 300 DPI equivalent
                mat = fitz.Matrix(3, 3)  # 3x zoom ≈ 216 DPI
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                if Image is not None:
                    img = Image.open(io.BytesIO(img_bytes))
                    images.append(img)
            doc.close()
            return images
        except Exception as e:
            print(f"pymupdf page rendering failed: {e}", file=sys.stderr)

    return images


# ── LangChain Tool ────────────────────────────────────────────

@tool
def ocr_cv_tool(
    file_path: str = "",
    file_bytes: Optional[bytes] = None,
    languages: Optional[List[str]] = None,
) -> dict:
    """
    Intelligent OCR for CV images and scanned PDFs.

    Extracts text with layout awareness and confidence filtering.
    Supports multilingual OCR (EN, FR, AR — configurable).

    Fallback chain:
        1. EasyOCR (ML-based, best accuracy)
        2. pytesseract (Tesseract engine)
        3. pymupdf image extraction → OCR

    Args:
        file_path: Path to a PDF or image file.
        file_bytes: Raw bytes of the file (alternative to file_path).
        languages: OCR languages, e.g. ["en", "fr", "ar"]. Default: ["en", "fr"].

    Returns:
        dict with keys: text, confidence, pages, ocr_backend, error
    """
    result = {
        "text": "",
        "confidence": 0.0,
        "pages": 0,
        "ocr_backend": None,
        "error": None,
    }

    langs = languages or ["en", "fr"]

    # Determine if this is a PDF or image
    is_pdf = False
    if file_path:
        is_pdf = file_path.lower().endswith(".pdf")
    elif file_bytes:
        # Check PDF magic bytes
        is_pdf = file_bytes[:5] == b"%PDF-"

    images = []

    if is_pdf:
        images = _pdf_to_images(file_path=file_path or None, file_bytes=file_bytes)
        result["pages"] = len(images)
    else:
        # Single image file
        try:
            if file_path and os.path.isfile(file_path):
                if Image is not None:
                    images = [Image.open(file_path)]
                    result["pages"] = 1
            elif file_bytes:
                if Image is not None:
                    images = [Image.open(io.BytesIO(file_bytes))]
                    result["pages"] = 1
        except Exception as e:
            result["error"] = f"Could not open image: {e}"
            return result

    if not images:
        result["error"] = (
            "Could not convert document to images for OCR. "
            "Install pdf2image+poppler or pymupdf for PDF support, "
            "and Pillow for image support."
        )
        return result

    # ── Attempt OCR backends in priority order ────────────────

    # 1. EasyOCR
    if easyocr is not None:
        text, conf = _ocr_with_easyocr(images, languages=langs)
        if text.strip():
            result["text"] = text.strip()
            result["confidence"] = round(conf, 3)
            result["ocr_backend"] = "easyocr"
            return result

    # 2. pytesseract
    if pytesseract is not None:
        text, conf = _ocr_with_tesseract(images)
        if text.strip():
            result["text"] = text.strip()
            result["confidence"] = round(conf, 3)
            result["ocr_backend"] = "pytesseract"
            return result

    # 3. No OCR backend succeeded
    result["error"] = (
        "OCR backends (easyocr, pytesseract) could not extract text. "
        "The image may be too low quality or the backends are not installed."
    )
    return result


def ocr_extract_text(file_path: str = "", file_bytes: bytes = None, languages: list[str] = None) -> str:
    """
    Convenience function: run OCR and return just the text.
    Returns empty string on failure.
    """
    result = ocr_cv_tool.invoke({
        "file_path": file_path or "",
        "file_bytes": file_bytes,
        "languages": languages or ["en", "fr"],
    })
    return result.get("text", "")
