"""RapidOCR wrapper: lazy engine loading and image-to-text helpers."""

from __future__ import annotations

import io

from PIL import Image

from config import OCR_TEXT_AREA_THRESHOLD, OCR_TEXT_WORD_THRESHOLD

_rapid_ocr_engine = None


def get_ocr_engine():
    """Return the module-level RapidOCR engine, loading it on first use.

    The import and model load are deferred because they take several
    seconds and are unnecessary for pure text-layer PDFs.

    Returns:
        RapidOCR: The shared OCR engine instance.
    """
    global _rapid_ocr_engine
    if _rapid_ocr_engine is None:
        print("    [Step] Loading RapidOCR engine...")
        from rapidocr_onnxruntime import RapidOCR
        _rapid_ocr_engine = RapidOCR()
        print("    [Step] RapidOCR engine ready")
    return _rapid_ocr_engine


def _extract_ocr_lines(result) -> tuple[list[str], float]:
    """Normalise RapidOCR output into text lines and a total text-box area.

    Args:
        result: Raw RapidOCR result — either a list of (box, text, score)
            entries or a (list, elapsed) tuple depending on version.

    Returns:
        tuple[list[str], float]: (text_lines, total_text_box_area_px).
    """
    lines: list[str] = []
    text_box_area = 0.0

    if not result:
        return lines, text_box_area

    if isinstance(result, tuple) and len(result) == 2:
        result = result[0]

    for line in result:
        if not line or len(line) < 2:
            continue
        box, text = line[0], line[1]
        lines.append(str(text))
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        text_box_area += (max(xs) - min(xs)) * (max(ys) - min(ys))

    return lines, text_box_area


def ocr_image_bytes(image_bytes: bytes) -> tuple[str, int, float]:
    """Run OCR on an encoded image and return text plus density metrics.

    Args:
        image_bytes (bytes): Encoded image data (PNG, JPEG, ...).

    Returns:
        tuple[str, int, float]: (full_text, word_count, text_area_ratio)
            where text_area_ratio is the fraction of the image covered by
            OCR text boxes. Returns ("", 0, 0.0) on any OCR failure.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        import numpy as np
        img_array = np.array(img)

        engine = get_ocr_engine()
        result, _elapsed = engine(img_array)
        lines, text_box_area = _extract_ocr_lines(result)

        full_text = "\n".join(lines)
        word_count = sum(len(line.split()) for line in lines)
        img_area = img.width * img.height
        text_area_ratio = text_box_area / img_area if img_area > 0 else 0.0

        return full_text, word_count, text_area_ratio
    except Exception as e:
        print(f"    [Step] OCR failed: {e}")
        return "", 0, 0.0


def ocr_page_image(pix) -> str:
    """OCR a rendered PDF page pixmap.

    Args:
        pix (fitz.Pixmap): Rasterised page image.

    Returns:
        str: Extracted text (may be empty).
    """
    print("    [Step] No text layer -> rendering page and running OCR...")
    img_bytes = pix.tobytes("png")
    text, word_count, _ = ocr_image_bytes(img_bytes)
    print(f"    [Step] OCR finished: {word_count} words extracted")
    return text


def is_text_dominant_ocr(word_count: int, text_area_ratio: float) -> bool:
    """Decide whether OCR output indicates a text-dominant image.

    Text-dominant images keep their OCR text as the description and skip
    the (slower, paid) vision-model call.

    Args:
        word_count (int): Number of words found by OCR.
        text_area_ratio (float): Fraction of image area covered by text boxes.

    Returns:
        bool: True if both thresholds from config are met.
    """
    return (word_count >= OCR_TEXT_WORD_THRESHOLD
            and text_area_ratio >= OCR_TEXT_AREA_THRESHOLD)
