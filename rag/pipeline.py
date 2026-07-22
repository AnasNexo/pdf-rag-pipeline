"""PDF ingestion orchestration: PDF → pages → blocks/tables/images → SQLite.

Per-page flow:

    PyMuPDF text layer dominant?
      ├── YES → store extracted text + typed semantic blocks
      │          (and describe the rendered page if visuals are present)
      └── NO  → render page → RapidOCR → store OCR text blocks
                 (vision-describe the page if OCR text is sparse)

    pdfplumber/Camelot → tables → markdown + semantic blocks
    Embedded images   → OCR-or-vision description → semantic blocks

Each page runs inside a SAVEPOINT so one failing page never loses the
rest of the document.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import traceback
from datetime import datetime, timezone

import fitz
import pdfplumber

from config import MIN_IMAGE_DIMENSION_PX, PAGE_RENDER_DPI
from rag.db import delete_document
from rag.loader import (
    build_ocr_blocks,
    extract_text_blocks,
    is_text_dominant_page,
    page_has_visual_content,
    page_vector_stats,
    render_page_bytes,
)
from rag.ocr import is_text_dominant_ocr, ocr_image_bytes, ocr_page_image
from rag.tables import extract_tables_from_page
from rag.utils import compute_file_hash, text_stats
from rag.vision import describe_image


# ---------------------------------------------------------------------------
# Row insertion helpers
# ---------------------------------------------------------------------------

def insert_semantic_blocks(conn: sqlite3.Connection, document_id: int,
                           page_id: int | None, image_id: int | None,
                           table_id: int | None, blocks: list[dict]) -> None:
    """Insert semantic block dicts into the semantic_blocks table.

    Args:
        conn (sqlite3.Connection): Open database connection.
        document_id (int): Owning document id.
        page_id (int | None): Owning page id, if any.
        image_id (int | None): Related image id, for image-description blocks.
        table_id (int | None): Related table id, for table blocks.
        blocks (list[dict]): Block dicts with keys block_index, block_type,
            source, bbox, text_content, confidence, and optionally
            section_title.

    Returns:
        None
    """
    if not blocks:
        return

    cur = conn.cursor()
    for block in blocks:
        cur.execute(
            """INSERT INTO semantic_blocks
               (document_id, page_id, image_id, table_id, block_index, block_type,
                source, bbox, text_content, confidence, section_title, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                page_id,
                image_id,
                table_id,
                block["block_index"],
                block["block_type"],
                block["source"],
                block.get("bbox"),
                block["text_content"],
                block.get("confidence"),
                block.get("section_title"),
                datetime.now(timezone.utc).isoformat(),
            )
        )


def insert_tables(conn: sqlite3.Connection, page_id: int, tables: list[dict],
                  document_id: int) -> list[int]:
    """Insert extracted table dicts into the tables table.

    Args:
        conn (sqlite3.Connection): Open database connection.
        page_id (int): Owning page id.
        tables (list[dict]): Table storage dicts from rag.tables.
        document_id (int): Owning document id.

    Returns:
        list[int]: Row ids of the inserted tables, in input order.
    """
    if not tables:
        return []

    import json
    cur = conn.cursor()
    table_ids = []
    for table in tables:
        cur.execute(
            """INSERT INTO tables
               (document_id, page_id, table_index, bbox, row_count, col_count,
                header_names, markdown, cells_json, source, accuracy)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                page_id,
                table["table_index"],
                table.get("bbox"),
                table.get("row_count"),
                table.get("col_count"),
                json.dumps(table.get("header_names", []), ensure_ascii=True),
                table["markdown"],
                table.get("cells_json"),
                table.get("source", "lines"),
                table.get("accuracy"),
            )
        )
        table_ids.append(cur.lastrowid)

    return table_ids


def _insert_image_with_block(conn: sqlite3.Connection, document_id: int,
                             page_id: int, image_index: int, bbox: str | None,
                             width: int | None, height: int | None,
                             description: str, description_source: str) -> int:
    """Insert an image row plus its matching image_description block.

    Args:
        conn (sqlite3.Connection): Open database connection.
        document_id (int): Owning document id.
        page_id (int): Owning page id.
        image_index (int): Position of the image on the page (0 for
            whole-page captures).
        bbox (str | None): Serialised bounding box, if known.
        width (int | None): Image width in pixels.
        height (int | None): Image height in pixels.
        description (str): OCR text or vision-model description.
        description_source (str): "ocr" or "vision".

    Returns:
        int: Row id of the inserted image.
    """
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO images
           (page_id, image_index, bbox, width, height, description, description_source)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (page_id, image_index, bbox, width, height, description, description_source)
    )
    image_id = cur.lastrowid
    insert_semantic_blocks(
        conn, document_id, page_id, image_id, None,
        [{
            "block_index": 0,
            "block_type": "image_description",
            "source": description_source,
            "bbox": bbox,
            "text_content": description,
            "confidence": None,
        }],
    )
    return image_id


# ---------------------------------------------------------------------------
# Page processing
# ---------------------------------------------------------------------------

def process_page(doc: "fitz.Document", page: "fitz.Page", plumb_page,
                 page_index: int, conn: sqlite3.Connection, document_id: int,
                 pdf_path: str | None = None, dpi: int = PAGE_RENDER_DPI) -> None:
    """Extract and store all content for a single PDF page.

    Stores the page text (text layer or OCR), typed semantic blocks,
    extracted tables, and image descriptions.

    Args:
        doc (fitz.Document): Open PyMuPDF document (for image extraction).
        page (fitz.Page): The page to process.
        plumb_page (pdfplumber.page.Page): Same page opened via pdfplumber
            (for table detection).
        page_index (int): 0-based page index.
        conn (sqlite3.Connection): Open database connection.
        document_id (int): Owning document id.
        pdf_path (str | None): Source PDF path (needed by Camelot).
        dpi (int): Render resolution for OCR / vision rasterisation.

    Returns:
        None

    Raises:
        RuntimeError: If the page INSERT yields no row id (broken FK).
    """
    cur = conn.cursor()

    print(f"  [Step] Checking page {page_index + 1} for a text layer (PyMuPDF)...")
    raw_text = page.get_text().strip()
    has_text = len(raw_text) > 0
    text_layer_dominant = is_text_dominant_page(raw_text) if has_text else False
    significant_drawings, drawing_coverage = page_vector_stats(page)
    has_visual_content = page_has_visual_content(page)

    if text_layer_dominant:
        source = "text"
        text_content = raw_text
        word_count, char_count = text_stats(raw_text)
        print(f"  [Step] Text layer found ({word_count} words, {char_count} chars) "
              "-> using extracted text")
        if has_visual_content:
            print(f"  [Step] Visual content detected alongside text "
                  f"(drawings={significant_drawings}, coverage={drawing_coverage:.0%})")
    else:
        source = "ocr"
        if has_text:
            word_count, char_count = text_stats(raw_text)
            print(f"  [Step] Text layer is sparse ({word_count} words, "
                  f"{char_count} chars) -> rendering page for OCR")
        else:
            print("  [Step] No text layer found -> rendering page for OCR")
        _, pix = render_page_bytes(page, dpi)
        text_content = ocr_page_image(pix)

    cur.execute(
        "INSERT INTO pages (document_id, page_number, source, text_content) "
        "VALUES (?, ?, ?, ?)",
        (document_id, page_index + 1, source, text_content)
    )
    page_id = cur.lastrowid
    if not page_id:
        raise RuntimeError(
            f"INSERT INTO pages returned lastrowid={page_id!r} "
            f"for document_id={document_id}, page={page_index + 1}. "
            "This usually means the document_id FK is broken or the INSERT failed."
        )
    print(f"  [Step] Page {page_index + 1} saved (id={page_id}, source={source})")

    # --- Semantic blocks ---------------------------------------------------
    if source == "text":
        raw_blocks = extract_text_blocks(page)
        # Carry the most recent heading forward as each block's section title.
        current_section: str | None = None
        semantic_blocks = []
        for block in raw_blocks:
            if block["block_type"] == "heading":
                current_section = " ".join(block["text_content"].split())
            block["section_title"] = current_section
            semantic_blocks.append(block)
        if not semantic_blocks and text_content.strip():
            semantic_blocks = [{
                "block_index": 0,
                "block_type": "text",
                "source": "text",
                "bbox": None,
                "text_content": text_content.strip(),
                "confidence": None,
                "section_title": None,
            }]
    else:
        semantic_blocks = build_ocr_blocks(text_content)
        for block in semantic_blocks:
            block["section_title"] = None

    insert_semantic_blocks(conn, document_id, page_id, None, None, semantic_blocks)
    if semantic_blocks:
        print(f"  [Step] Saved {len(semantic_blocks)} semantic block(s) "
              f"for page {page_index + 1}")

    # --- Tables -------------------------------------------------------------
    extracted_tables = extract_tables_from_page(plumb_page, pdf_path, page_index)
    if extracted_tables:
        table_ids = insert_tables(conn, page_id, extracted_tables, document_id)
        print(f"  [Step] Saved {len(table_ids)} table(s) for page {page_index + 1}")

        for table, table_id in zip(extracted_tables, table_ids):
            insert_semantic_blocks(
                conn, document_id, page_id, None, table_id,
                [{
                    "block_index": 0,
                    "block_type": "table",
                    "source": table.get("source", "lines"),
                    "bbox": table.get("bbox"),
                    "text_content": table["markdown"],
                    "confidence": None,
                }],
            )

    # --- Images -------------------------------------------------------------
    print(f"  [Step] Detecting image regions on page {page_index + 1}...")
    image_list = page.get_images(full=True)
    print(f"  [Step] Found {len(image_list)} embedded image(s)")

    # OCR pages and visually-rich text pages are captured as one whole-page
    # image; otherwise each embedded image is described individually.
    should_analyze_whole_page = source == "ocr" or has_visual_content

    if should_analyze_whole_page:
        if source == "ocr" and len(image_list) == 0:
            print("  [Step] No embedded images and no text layer "
                  "-> whole-page raster fallback...")
        elif source == "text" and has_visual_content:
            print("  [Step] Text layer present but visual content detected "
                  "-> describing rendered page...")

        image_bytes, pix = render_page_bytes(page, dpi)

        if source == "text":
            # Page text already captured above; skip re-OCR and go straight
            # to the vision model for the visuals.
            description = describe_image(image_bytes, context_text=text_content)
            description_source = "vision"
        else:
            ocr_text, word_count, text_area_ratio = ocr_image_bytes(image_bytes)
            print(f"  [Step] Whole-page OCR: {word_count} words, "
                  f"text_area_ratio={text_area_ratio:.0%}")
            if is_text_dominant_ocr(word_count, text_area_ratio):
                description = ocr_text
                description_source = "ocr"
            else:
                description = describe_image(image_bytes, context_text=text_content)
                description_source = "vision"
                if ocr_text.strip():
                    description += f"\n\n[OCR text in image]:\n{ocr_text.strip()}"

        _insert_image_with_block(conn, document_id, page_id, 0, None,
                                 pix.width, pix.height, description,
                                 description_source)
        print(f"  [Step] Whole-page image saved (source={description_source})")
        return

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            width = base_image.get("width")
            height = base_image.get("height")
        except Exception as e:
            print(f"    [Step] Image {img_index}: failed to extract ({e}), skipping")
            continue

        bbox = None
        try:
            rects = page.get_image_rects(xref)
            if rects:
                r = rects[0]
                bbox = f"{r.x0:.1f},{r.y0:.1f},{r.x1:.1f},{r.y1:.1f}"
        except Exception:
            pass

        if width and height and (width < MIN_IMAGE_DIMENSION_PX
                                 or height < MIN_IMAGE_DIMENSION_PX):
            print(f"    [Step] Image {img_index}: too small ({width}x{height}), skipping")
            continue

        print(f"    [Step] Image {img_index} ({width}x{height}): running OCR first...")
        ocr_text, word_count, text_area_ratio = ocr_image_bytes(image_bytes)
        print(f"    [Step] Image {img_index}: {word_count} words, "
              f"text_area_ratio={text_area_ratio:.0%}")

        if is_text_dominant_ocr(word_count, text_area_ratio):
            description = ocr_text
            description_source = "ocr"
            print(f"    [Step] Image {img_index}: text-dominant -> OCR only, "
                  "skipping vision model")
        else:
            print(f"    [Step] Image {img_index}: sending to vision model")
            description = describe_image(image_bytes, context_text=text_content)
            description_source = "vision"
            if ocr_text.strip():
                description += f"\n\n[OCR text in image]:\n{ocr_text.strip()}"

        _insert_image_with_block(conn, document_id, page_id, img_index, bbox,
                                 width, height, description, description_source)
        print(f"    [Step] Image {img_index}: saved (source={description_source})")


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def process_pdf(filepath: str, conn: sqlite3.Connection, force: bool = False) -> None:
    """Ingest a single PDF into the database.

    Skips files whose content hash already exists unless *force* is set,
    in which case the previous data is deleted and the file reprocessed.

    Args:
        filepath (str): Path to the PDF file.
        conn (sqlite3.Connection): Open database connection with the
            ingestion schema initialised.
        force (bool): Reprocess even if the content hash already exists.

    Returns:
        None

    Raises:
        RuntimeError: If the document row cannot be inserted.
    """
    cur = conn.cursor()
    filename = os.path.basename(filepath)
    content_hash = compute_file_hash(filepath)

    cur.execute(
        "SELECT id, filename, processed_at FROM documents WHERE content_hash = ?",
        (content_hash,)
    )
    existing = cur.fetchone()

    if existing:
        existing_id, existing_filename, processed_at = existing
        if not force:
            print(f"Skipping '{filename}': already processed as '{existing_filename}' "
                  f"(doc id={existing_id}, on {processed_at}). Use --force to reprocess.")
            return
        print(f"'{filename}' already processed (doc id={existing_id}). "
              f"--force: deleting old data and reprocessing.")
        delete_document(conn, existing_id)

    doc = fitz.open(filepath)
    page_count = doc.page_count

    cur.execute(
        "INSERT INTO documents (filename, filepath, content_hash, page_count, processed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (filename, filepath, content_hash, page_count,
         datetime.now(timezone.utc).isoformat())
    )
    document_id = cur.lastrowid
    # Commit the document row BEFORE processing pages so per-page
    # savepoint rollbacks can never take the parent row with them.
    conn.commit()

    if not document_id:
        raise RuntimeError(f"Failed to insert document row for '{filename}'")

    print(f"\nProcessing '{filename}' ({page_count} pages)... [doc id={document_id}]")

    with pdfplumber.open(filepath) as plumb_doc:
        for i, page in enumerate(doc):
            print(f"  Page {i + 1}/{page_count}")
            try:
                conn.execute("SAVEPOINT page_proc")
                plumb_page = plumb_doc.pages[i]
                process_page(doc, page, plumb_page, i, conn, document_id, filepath)
                conn.execute("RELEASE SAVEPOINT page_proc")
            except Exception as e:
                conn.execute("ROLLBACK TO SAVEPOINT page_proc")
                conn.execute("RELEASE SAVEPOINT page_proc")
                print(f"  [ERROR] Page {i + 1} failed: {e}", file=sys.stderr)
                traceback.print_exc()

    # Commit all page data. The original script relied on a later commit
    # (or none at all for the final document) — closing a connection with
    # a pending transaction rolls it back, silently dropping pages.
    conn.commit()
    doc.close()
    print(f"Done with '{filename}'.\n")
