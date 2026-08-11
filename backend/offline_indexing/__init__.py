"""Offline normalization and baseline retrieval for organizer-provided artifacts."""

from .artifact_indexer import build_index
from .vector_index import VectorIndex

__all__ = ["VectorIndex", "build_index"]
