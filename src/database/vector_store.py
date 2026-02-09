"""
Abstract vector store interface for speaker embeddings.

Provides a consistent API for different vector database backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SearchResult:
    """Vector search result."""

    id: str
    score: float
    payload: Dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "score": round(self.score, 4),
            "payload": self.payload,
        }


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store and create collection if needed."""
        pass

    @abstractmethod
    async def insert(
        self,
        id: str,
        vector: np.ndarray,
        payload: Dict[str, Any],
    ) -> None:
        """
        Insert a vector with payload.

        Args:
            id: Unique identifier for the vector
            vector: Embedding vector
            payload: Associated metadata
        """
        pass

    @abstractmethod
    async def insert_batch(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """
        Insert multiple vectors.

        Args:
            ids: List of unique identifiers
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries
        """
        pass

    @abstractmethod
    async def search(
        self,
        vector: np.ndarray,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            vector: Query vector
            limit: Maximum number of results
            filter: Optional filter conditions
            score_threshold: Minimum similarity score

        Returns:
            List of SearchResult sorted by similarity
        """
        pass

    @abstractmethod
    async def get(
        self,
        id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get vector and payload by ID.

        Args:
            id: Vector ID

        Returns:
            Dictionary with vector and payload, or None if not found
        """
        pass

    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Delete vectors by ID or filter.

        Args:
            ids: List of IDs to delete
            filter: Filter conditions for deletion

        Returns:
            Number of deleted vectors
        """
        pass

    @abstractmethod
    async def count(
        self,
        filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Count vectors in the store.

        Args:
            filter: Optional filter conditions

        Returns:
            Number of vectors
        """
        pass

    @abstractmethod
    async def get_all_by_filter(
        self,
        filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Get all vectors matching a filter.

        Args:
            filter: Filter conditions

        Returns:
            List of dictionaries with vector and payload
        """
        pass
