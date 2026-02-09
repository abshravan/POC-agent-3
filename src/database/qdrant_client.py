"""
Qdrant vector database implementation.

Provides high-performance vector search using Qdrant.
"""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from config.settings import get_settings
from config.logging_config import get_logger
from src.database.vector_store import VectorStore, SearchResult

logger = get_logger(__name__)


class QdrantVectorStore(VectorStore):
    """
    Qdrant vector store implementation.

    Supports both embedded mode (local) and server mode.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        path: Optional[str] = None,
        embedding_dim: int = 192,
    ):
        """
        Initialize Qdrant client.

        Args:
            collection_name: Name of the collection
            host: Qdrant server host (None for embedded mode)
            port: Qdrant server port
            path: Path for embedded mode storage
            embedding_dim: Dimension of embeddings
        """
        settings = get_settings()

        self.collection_name = collection_name or settings.qdrant_collection_name
        self.embedding_dim = embedding_dim

        # Determine connection mode
        self.path = path or settings.qdrant_path
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port

        # Initialize client (prefer embedded mode for local deployment)
        if self.path:
            logger.info(
                "qdrant_embedded_mode",
                path=self.path,
                collection=self.collection_name,
            )
            self.client = QdrantClient(path=self.path)
        else:
            logger.info(
                "qdrant_server_mode",
                host=self.host,
                port=self.port,
                collection=self.collection_name,
            )
            self.client = QdrantClient(host=self.host, port=self.port)

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize collection if it doesn't exist."""
        if self._initialized:
            return

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                # Create collection with cosine similarity
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.embedding_dim,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                    # Optimize for small-medium collections
                    hnsw_config=qdrant_models.HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10000,
                    ),
                )
                logger.info(
                    "collection_created",
                    collection=self.collection_name,
                    dim=self.embedding_dim,
                )
            else:
                logger.info(
                    "collection_exists",
                    collection=self.collection_name,
                )

            self._initialized = True

        except Exception as e:
            logger.error("qdrant_init_error", error=str(e))
            raise

    async def insert(
        self,
        id: str,
        vector: np.ndarray,
        payload: Dict[str, Any],
    ) -> None:
        """Insert a single vector."""
        await self.initialize()

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=id,
                        vector=vector.tolist(),
                        payload=payload,
                    )
                ],
            )
            logger.debug("vector_inserted", id=id)

        except Exception as e:
            logger.error("insert_error", id=id, error=str(e))
            raise

    async def insert_batch(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """Insert multiple vectors."""
        await self.initialize()

        try:
            points = [
                qdrant_models.PointStruct(
                    id=id,
                    vector=vector.tolist(),
                    payload=payload,
                )
                for id, vector, payload in zip(ids, vectors, payloads)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.debug("batch_inserted", count=len(ids))

        except Exception as e:
            logger.error("batch_insert_error", count=len(ids), error=str(e))
            raise

    async def search(
        self,
        vector: np.ndarray,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        await self.initialize()

        try:
            # Convert filter to Qdrant format
            qdrant_filter = None
            if filter:
                qdrant_filter = self._build_filter(filter)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector.tolist(),
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
            )

            return [
                SearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {},
                )
                for r in results
            ]

        except Exception as e:
            logger.error("search_error", error=str(e))
            raise

    async def get(
        self,
        id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get vector and payload by ID."""
        await self.initialize()

        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[id],
                with_vectors=True,
                with_payload=True,
            )

            if not results:
                return None

            point = results[0]
            return {
                "id": str(point.id),
                "vector": np.array(point.vector),
                "payload": point.payload or {},
            }

        except Exception as e:
            logger.error("get_error", id=id, error=str(e))
            return None

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Delete vectors by ID or filter."""
        await self.initialize()

        try:
            if ids:
                # Delete by IDs
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=qdrant_models.PointIdsList(
                        points=ids,
                    ),
                )
                return len(ids)

            elif filter:
                # Delete by filter
                qdrant_filter = self._build_filter(filter)
                # First count matching points
                count_before = await self.count(filter)

                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_filter,
                    ),
                )
                return count_before

            return 0

        except Exception as e:
            logger.error("delete_error", error=str(e))
            raise

    async def count(
        self,
        filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Count vectors in collection."""
        await self.initialize()

        try:
            if filter:
                qdrant_filter = self._build_filter(filter)
                result = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=qdrant_filter,
                    exact=True,
                )
            else:
                result = self.client.count(
                    collection_name=self.collection_name,
                    exact=True,
                )

            return result.count

        except Exception as e:
            logger.error("count_error", error=str(e))
            return 0

    async def get_all_by_filter(
        self,
        filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Get all vectors matching a filter."""
        await self.initialize()

        try:
            qdrant_filter = self._build_filter(filter)

            # Scroll through all matching points
            results = []
            offset = None

            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_filter,
                    limit=100,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True,
                )

                for point in points:
                    results.append({
                        "id": str(point.id),
                        "vector": np.array(point.vector) if point.vector else None,
                        "payload": point.payload or {},
                    })

                if offset is None:
                    break

            return results

        except Exception as e:
            logger.error("get_all_error", error=str(e))
            return []

    def _build_filter(
        self,
        filter_dict: Dict[str, Any],
    ) -> qdrant_models.Filter:
        """
        Build Qdrant filter from dictionary.

        Supports simple key-value matching.

        Args:
            filter_dict: Filter conditions

        Returns:
            Qdrant Filter object
        """
        conditions = []

        for key, value in filter_dict.items():
            if isinstance(value, bool):
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value),
                    )
                )
            elif isinstance(value, (int, float)):
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value),
                    )
                )
            elif isinstance(value, str):
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value),
                    )
                )
            elif isinstance(value, list):
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchAny(any=value),
                    )
                )

        return qdrant_models.Filter(must=conditions)

    async def close(self) -> None:
        """Close the client connection."""
        if self.client:
            self.client.close()
            logger.info("qdrant_closed")
