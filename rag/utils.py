"""Small shared helpers used across pipeline stages."""

from __future__ import annotations

import hashlib


def compute_file_hash(filepath: str) -> str:
    """Compute the SHA-256 hash of a file's contents.

    Args:
        filepath (str): Path to the file to hash.

    Returns:
        str: Hex-encoded SHA-256 digest.

    Raises:
        OSError: If the file cannot be opened or read.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def text_stats(text: str) -> tuple[int, int]:
    """Count words and characters in a normalised text string.

    Args:
        text (str): Raw text (may be None or empty).

    Returns:
        tuple[int, int]: (word_count, char_count) of the stripped text.
    """
    normalized = (text or "").strip()
    return len(normalized.split()), len(normalized)


def bbox_to_text(bbox) -> str | None:
    """Serialise a 4-tuple bounding box to a compact comma-separated string.

    Args:
        bbox: Iterable of four numbers (x0, y0, x1, y1), or a falsy value.

    Returns:
        str | None: "x0,y0,x1,y1" with one decimal place, or None if the
            input is falsy or malformed.
    """
    if not bbox:
        return None
    try:
        x0, y0, x1, y1 = bbox
        return f"{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}"
    except Exception:
        return None
