"""Table extraction from PDF pages.

Runs four complementary detection strategies in confidence order and
deduplicates the results by markdown content:

    1. pdfplumber lines/lines      — full horizontal+vertical line grids
    2. pdfplumber lines_strict     — rect-bordered tables
    3. Camelot lattice             — fully-bordered tables
    4. Camelot stream              — whitespace/rule-aligned tables,
                                     gated by Camelot's accuracy score

Every candidate grid passes through validation heuristics that reject
paragraph text masquerading as a table.
"""

from __future__ import annotations

import hashlib
import json

import camelot

from config import (
    CAMELOT_STREAM_EDGE_TOL,
    CAMELOT_STREAM_MIN_ACCURACY,
    CAMELOT_STREAM_ROW_TOL,
    PDFPLUMBER_JOIN_TOLERANCE,
    PDFPLUMBER_SNAP_TOLERANCE,
    TABLE_MAX_AVG_CELL_CHARS,
    TABLE_MAX_CELL_CHARS,
    TABLE_MAX_EMPTY_CELL_RATIO,
    TABLE_MAX_INCONSISTENT_ROW_RATIO,
)


def table_grid_to_markdown(grid: list[list[str | None]]) -> str:
    """Render a 2-D cell grid as a GitHub-style markdown table.

    Cell newlines become <br> and vertical bars are escaped so they do
    not break markdown columns. Rows are padded/trimmed to the header's
    column count.

    Args:
        grid (list[list[str | None]]): Rows of cell values; row 0 is the
            header.

    Returns:
        str: Markdown table, or "" if the grid is empty.
    """
    if not grid or not grid[0]:
        return ""

    cols = len(grid[0])

    formatted_grid = []
    for row in grid:
        formatted_row = []
        for cell in row:
            val = str(cell or "").strip()
            val = val.replace("\n", "<br>")
            val = val.replace("|", "&#124;")
            formatted_row.append(val)
        if len(formatted_row) < cols:
            formatted_row += [""] * (cols - len(formatted_row))
        elif len(formatted_row) > cols:
            formatted_row = formatted_row[:cols]
        formatted_grid.append(formatted_row)

    header = "| " + " | ".join(formatted_grid[0]) + " |"
    separator = "| " + " | ".join(["---"] * cols) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in formatted_grid[1:]]

    return "\n".join([header, separator] + rows)


def _get_page_lines(plumb_page) -> tuple[list, list]:
    """Separate a page's drawing lines into horizontal and vertical sets.

    Args:
        plumb_page (pdfplumber.page.Page): Page to analyse.

    Returns:
        tuple[list, list]: (horizontal_lines, vertical_lines) as
            pdfplumber line dicts.
    """
    h_lines, v_lines = [], []
    for line in plumb_page.lines:
        horiz = line["x1"] - line["x0"]
        vert = abs(line.get("bottom", line.get("y1", 0)) - line.get("top", line.get("y0", 0)))
        if horiz >= 20 and vert < 3:
            h_lines.append(line)
        elif horiz < 3 and vert >= 15:
            v_lines.append(line)
    return h_lines, v_lines


def _lines_form_grid(h_lines: list, v_lines: list, tol: float = 5.0) -> bool:
    """Check whether any horizontal/vertical line pair intersects.

    An intersection indicates a real table grid rather than decorative
    rules.

    Args:
        h_lines (list): Horizontal line dicts.
        v_lines (list): Vertical line dicts.
        tol (float): Intersection tolerance in points.

    Returns:
        bool: True if at least one H/V pair intersects.
    """
    for h in h_lines:
        h_y = (h.get("top", 0) + h.get("bottom", h.get("top", 0))) / 2
        for v in v_lines:
            v_top = min(v.get("top", 0), v.get("bottom", 0))
            v_bot = max(v.get("top", 0), v.get("bottom", 0))
            if (h["x0"] - tol <= v["x0"] <= h["x1"] + tol
                    and v_top - tol <= h_y <= v_bot + tol):
                return True
    return False


def _validate_table_grid(grid) -> bool:
    """Heuristically decide whether a grid is a real table.

    Rejects grids that are too small, single-column, mostly empty, or
    contain paragraph-length cells.

    Args:
        grid: Rows of cell values as extracted by pdfplumber/Camelot.

    Returns:
        bool: True only if the grid looks like an actual table.
    """
    if not grid or len(grid) < 2:
        return False
    col_count = max(len(row) for row in grid)
    if col_count < 2:
        return False
    inconsistent = sum(1 for row in grid if abs(len(row) - col_count) > 1)
    if inconsistent > len(grid) * TABLE_MAX_INCONSISTENT_ROW_RATIO:
        return False
    all_cells = [str(cell or "").strip() for row in grid for cell in row]
    nonempty = [c for c in all_cells if c]
    if not nonempty:
        return False
    if (len(all_cells) - len(nonempty)) / len(all_cells) > TABLE_MAX_EMPTY_CELL_RATIO:
        return False
    avg_len = sum(len(c) for c in nonempty) / len(nonempty)
    if avg_len > TABLE_MAX_AVG_CELL_CHARS:
        return False
    if max(len(c) for c in nonempty) > TABLE_MAX_CELL_CHARS:
        return False
    return True


def _plumb_table_to_dict(table, table_index: int, source: str) -> dict | None:
    """Convert a pdfplumber Table into the pipeline's storage dict.

    Args:
        table (pdfplumber.table.Table): Detected table object.
        table_index (int): Position of this table on the page.
        source (str): Detection strategy label (e.g. "pdfplumber_grid").

    Returns:
        dict | None: Storage dict, or None if extraction fails validation.
    """
    try:
        grid = table.extract()
    except Exception:
        return None
    if not _validate_table_grid(grid):
        return None
    markdown = table_grid_to_markdown(grid)
    if not markdown or len(markdown.splitlines()) < 3:
        return None
    bbox = None
    try:
        if table.bbox:
            b = table.bbox
            bbox = f"{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}"
    except Exception:
        pass
    return {
        "table_index": table_index,
        "bbox": bbox,
        "row_count": len(grid),
        "col_count": max(len(row) for row in grid),
        "header_names": [str(x or "").strip() for x in grid[0]],
        "markdown": markdown,
        "cells_json": json.dumps(grid, ensure_ascii=True),
        "source": source,
    }


def _camelot_table_to_dict(table, table_index: int, source: str) -> dict | None:
    """Convert a Camelot Table into the pipeline's storage dict.

    Args:
        table (camelot.core.Table): Detected table object.
        table_index (int): Position of this table on the page.
        source (str): Detection strategy label (e.g. "camelot_lattice").

    Returns:
        dict | None: Storage dict (including Camelot's accuracy score),
            or None if extraction fails validation.
    """
    if table.df.empty or len(table.df) < 2:
        return None
    grid = table.df.values.tolist()
    if not _validate_table_grid(grid):
        return None
    markdown = table_grid_to_markdown(grid)
    if not markdown or len(markdown.splitlines()) < 3:
        return None
    try:
        b = table.bbox
        bbox = f"{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}"
    except Exception:
        bbox = None
    return {
        "table_index": table_index,
        "bbox": bbox,
        "row_count": len(grid),
        "col_count": len(grid[0]) if grid else 0,
        "header_names": [str(x or "").strip() for x in grid[0]],
        "markdown": markdown,
        "cells_json": json.dumps(grid, ensure_ascii=True),
        "source": source,
        "accuracy": round(float(table.accuracy), 2),
    }


def extract_tables_from_page(plumb_page, pdf_path: str | None = None,
                             page_num: int | None = None) -> list[dict]:
    """Detect and extract all tables on one page using layered strategies.

    Results from all strategies are merged and deduplicated by markdown
    content hash, keeping the first (highest-confidence) occurrence.

    Args:
        plumb_page (pdfplumber.page.Page): Page to scan.
        pdf_path (str | None): Path to the source PDF — required for the
            Camelot strategies, which re-open the file themselves.
        page_num (int | None): 0-based page index, required for Camelot.

    Returns:
        list[dict]: Unique table storage dicts with sequential
            table_index values.
    """
    results: list[dict] = []
    seen_hashes: set[str] = set()

    def _add(t_dict: dict | None) -> None:
        if t_dict is None:
            return
        h = hashlib.md5(t_dict["markdown"].encode()).hexdigest()
        if h in seen_hashes:
            return
        seen_hashes.add(h)
        t_dict["table_index"] = len(results)
        results.append(t_dict)

    h_lines, v_lines = _get_page_lines(plumb_page)
    has_grid = _lines_form_grid(h_lines, v_lines)

    # Strategy 1: full H+V grid — highest confidence.
    if has_grid:
        try:
            for t in plumb_page.find_tables(table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": PDFPLUMBER_SNAP_TOLERANCE,
                "join_tolerance": PDFPLUMBER_JOIN_TOLERANCE,
                "min_words_vertical": 0,
                "min_words_horizontal": 0,
            }):
                _add(_plumb_table_to_dict(t, len(results), "pdfplumber_grid"))
        except Exception as e:
            print(f"    [Step] Strategy 1 (grid lines) failed: {e}")

    # Strategy 2: rect-bordered tables.
    if plumb_page.rects:
        try:
            for t in plumb_page.find_tables(table_settings={
                "vertical_strategy": "lines_strict",
                "horizontal_strategy": "lines_strict",
                "snap_tolerance": PDFPLUMBER_SNAP_TOLERANCE,
                "join_tolerance": PDFPLUMBER_JOIN_TOLERANCE,
            }):
                _add(_plumb_table_to_dict(t, len(results), "pdfplumber_rects"))
        except Exception as e:
            print(f"    [Step] Strategy 2 (rects) failed: {e}")

    # Strategy 3: Camelot lattice — fully-bordered tables.
    if pdf_path and page_num is not None:
        try:
            lattice_tables = camelot.read_pdf(
                pdf_path, pages=str(page_num + 1),
                flavor="lattice", suppress_stdout=True,
            )
            print(f"    [Camelot lattice] Found {len(lattice_tables)} candidate tables")
            for t in lattice_tables:
                _add(_camelot_table_to_dict(t, len(results), "camelot_lattice"))
        except Exception as e:
            print(f"    [Camelot lattice] Failed: {e}")

    # Strategy 4: Camelot stream — Camelot's accuracy score acts as the
    # false-positive guard for borderless tables.
    if pdf_path and page_num is not None:
        try:
            stream_tables = camelot.read_pdf(
                pdf_path, pages=str(page_num + 1),
                flavor="stream", suppress_stdout=True,
                edge_tol=CAMELOT_STREAM_EDGE_TOL, row_tol=CAMELOT_STREAM_ROW_TOL,
            )
            print(f"    [Camelot stream] Found {len(stream_tables)} candidate tables")
            for t in stream_tables:
                if t.accuracy < CAMELOT_STREAM_MIN_ACCURACY:
                    continue
                _add(_camelot_table_to_dict(t, len(results), "camelot_stream"))
        except Exception as e:
            print(f"    [Camelot stream] Failed: {e}")

    print(f"  [Step] Final unique tables extracted: {len(results)}")
    return results
