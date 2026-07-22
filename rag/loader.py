"""Page-level PDF analysis and semantic-block extraction (PyMuPDF).

Decides whether a page's text layer is trustworthy or OCR is needed,
detects visual content (embedded images, vector drawings), renders pages
to images, and converts a page into typed *semantic blocks* (text /
heading / ocr_text) with heading detection based on relative font size.
"""

from __future__ import annotations

import fitz

from config import (
    BOLD_HEADING_MAX_WORDS,
    BOLD_HEADING_MIN_SIZE_RATIO,
    HEADING_MAX_WORDS,
    HEADING_SIZE_RATIO,
    SEMANTIC_BLOCK_MAX_CHARS,
    SIGNIFICANT_DRAWING_AREA_RATIO,
    TEXT_LAYER_CHAR_THRESHOLD,
    TEXT_LAYER_WORD_THRESHOLD,
    VECTOR_DRAWING_COUNT_THRESHOLD,
    VECTOR_DRAWING_COVERAGE_THRESHOLD,
)
from rag.utils import bbox_to_text, text_stats


def is_text_dominant_page(text: str) -> bool:
    """Decide whether a page's embedded text layer is substantial.

    Pages failing this check are rasterised and OCR'd instead.

    Args:
        text (str): Raw text extracted from the page's text layer.

    Returns:
        bool: True if both word- and char-count thresholds are met.
    """
    word_count, char_count = text_stats(text)
    return (word_count >= TEXT_LAYER_WORD_THRESHOLD
            and char_count >= TEXT_LAYER_CHAR_THRESHOLD)


def render_page_bytes(page: "fitz.Page", dpi: int) -> tuple[bytes, "fitz.Pixmap"]:
    """Rasterise a PDF page to a PNG image at the given resolution.

    Args:
        page (fitz.Page): Page to render.
        dpi (int): Target resolution in dots per inch.

    Returns:
        tuple[bytes, fitz.Pixmap]: (png_bytes, pixmap).
    """
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return pix.tobytes("png"), pix


def page_vector_stats(page: "fitz.Page") -> tuple[int, float]:
    """Measure vector-drawing activity on a page.

    Used to detect charts/diagrams drawn with vector primitives, which
    have no embedded raster image to inspect.

    Args:
        page (fitz.Page): Page to analyse.

    Returns:
        tuple[int, float]: (significant_drawing_count, coverage_ratio)
            where coverage_ratio is the fraction of page area covered by
            drawings, capped at 1.0.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return 0, 0.0

    page_area = float(page.rect.width * page.rect.height)
    if page_area <= 0:
        return 0, 0.0

    significant_drawings = 0
    covered_area = 0.0

    for drawing in drawings:
        rect = drawing.get("rect")
        if not rect:
            continue
        rect_area = max(0.0, rect.width) * max(0.0, rect.height)
        if rect_area <= 0:
            continue
        covered_area += rect_area
        if rect_area / page_area >= SIGNIFICANT_DRAWING_AREA_RATIO:
            significant_drawings += 1

    return significant_drawings, min(1.0, covered_area / page_area)


def page_has_visual_content(page: "fitz.Page") -> bool:
    """Decide whether a page contains images or significant vector graphics.

    Args:
        page (fitz.Page): Page to analyse.

    Returns:
        bool: True if the page has embedded images, enough significant
            vector drawings, or enough drawing coverage (per config).
    """
    embedded_images = page.get_images(full=True)
    significant_drawings, drawing_coverage = page_vector_stats(page)
    return (bool(embedded_images)
            or significant_drawings >= VECTOR_DRAWING_COUNT_THRESHOLD
            or drawing_coverage >= VECTOR_DRAWING_COVERAGE_THRESHOLD)


def chunk_text_for_blocks(text: str, max_chars: int = SEMANTIC_BLOCK_MAX_CHARS) -> list[str]:
    """Split free text into paragraph-aligned blocks of bounded size.

    Paragraphs are packed greedily up to *max_chars*; a single paragraph
    longer than the limit is hard-split into max_chars windows.

    Args:
        text (str): Text to split (may be None or empty).
        max_chars (int): Maximum characters per block.

    Returns:
        list[str]: Non-empty text blocks in original order.
    """
    normalized = (text or "").strip()
    if not normalized:
        return []

    paragraphs = [part.strip() for part in normalized.splitlines() if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append("\n".join(current).strip())
                current = []
                current_length = 0
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start:start + max_chars].strip())
                start += max_chars
            continue

        extra_length = len(paragraph) + (1 if current else 0)
        if current and current_length + extra_length > max_chars:
            chunks.append("\n".join(current).strip())
            current = [paragraph]
            current_length = len(paragraph)
        else:
            current.append(paragraph)
            current_length += extra_length

    if current:
        chunks.append("\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def extract_text_blocks(page: "fitz.Page") -> list[dict]:
    """Extract typed semantic blocks from a page's text layer.

    Blocks are classified as "heading" when their average font size
    exceeds the page's median body size (or they are short and bold);
    everything else is "text".

    Args:
        page (fitz.Page): Page with a usable text layer.

    Returns:
        list[dict]: Block dicts with keys block_index, block_type, source,
            bbox, text_content, confidence. Empty on extraction failure.
    """
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return []

    # Median body font size lets headings be detected by *relative* size,
    # which is robust across documents with different base font sizes.
    all_sizes = []
    for b in text_dict.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size", 0)
                if size > 0 and span.get("text", "").strip():
                    all_sizes.append(size)
    all_sizes.sort()
    median_size = all_sizes[len(all_sizes) // 2] if all_sizes else 12.0

    blocks = []
    for block_index, block in enumerate(text_dict.get("blocks", [])):
        if block.get("type") != 0:
            continue

        lines = []
        span_sizes = []
        has_bold = False
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = " ".join(
                span.get("text", "").strip() for span in spans
                if span.get("text", "").strip()
            ).strip()
            if line_text:
                lines.append(line_text)
            for span in spans:
                if span.get("text", "").strip():
                    span_sizes.append(span.get("size", 0))
                    if span.get("flags", 0) & 16:  # bit 4 = bold
                        has_bold = True

        block_text = "\n".join(lines).strip()
        if not block_text:
            continue

        block_type = "text"
        if span_sizes:
            avg_size = sum(span_sizes) / len(span_sizes)
            word_count = len(block_text.split())
            if avg_size > median_size * HEADING_SIZE_RATIO and word_count <= HEADING_MAX_WORDS:
                block_type = "heading"
            elif (has_bold and word_count <= BOLD_HEADING_MAX_WORDS
                  and avg_size >= median_size * BOLD_HEADING_MIN_SIZE_RATIO):
                block_type = "heading"

        blocks.append({
            "block_index": block_index,
            "block_type": block_type,
            "source": "text",
            "bbox": bbox_to_text(block.get("bbox")),
            "text_content": block_text,
            "confidence": None,
        })

    return blocks


def build_ocr_blocks(text_content: str) -> list[dict]:
    """Wrap OCR page text into semantic block dicts of bounded size.

    Args:
        text_content (str): Full OCR text for one page.

    Returns:
        list[dict]: Block dicts with block_type "ocr_text" and no bbox.
    """
    return [
        {
            "block_index": block_index,
            "block_type": "ocr_text",
            "source": "ocr",
            "bbox": None,
            "text_content": chunk,
            "confidence": None,
        }
        for block_index, chunk in enumerate(chunk_text_for_blocks(text_content))
    ]
