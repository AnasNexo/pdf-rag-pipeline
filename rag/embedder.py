"""Sentence-transformers model cache and vector (de)serialisation.

Vectors are L2-normalised at encode time so a dot product between two
stored vectors equals their cosine similarity. They are persisted as
packed little-endian float32 blobs in SQLite.
"""

from __future__ import annotations

import struct

from config import EMBEDDING_MODEL

_model_cache: dict[str, "SentenceTransformer"] = {}


def get_model(model_name: str = EMBEDDING_MODEL):
    """Return a cached SentenceTransformer, loading it on first use.

    Args:
        model_name (str): Hugging Face model name.

    Returns:
        SentenceTransformer: The cached model instance.
    """
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading model '{model_name}'...")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def encode_texts(texts: list[str], model_name: str = EMBEDDING_MODEL):
    """Encode texts into L2-normalised embedding vectors.

    Args:
        texts (list[str]): Texts to embed.
        model_name (str): Hugging Face model name.

    Returns:
        numpy.ndarray: Array of shape (len(texts), dim).
    """
    model = get_model(model_name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def pack_vector(vec: list[float]) -> bytes:
    """Serialise a float vector to a packed float32 blob.

    Args:
        vec (list[float]): Vector values.

    Returns:
        bytes: Packed binary representation (4 bytes per value).
    """
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    """Deserialise a packed float32 blob back into a float list.

    Args:
        blob (bytes): Blob produced by pack_vector.

    Returns:
        list[float]: Vector values.
    """
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))
