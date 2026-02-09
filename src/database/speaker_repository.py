"""
Speaker repository for managing speaker metadata.

Handles CRUD operations for speaker records.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Speaker:
    """Speaker entity."""

    id: str
    employee_id: str
    name: str
    department: Optional[str] = None
    enrolled_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    embedding_count: int = 0
    consistency_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "name": self.name,
            "department": self.department,
            "enrolled_at": self.enrolled_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
            "embedding_count": self.embedding_count,
            "consistency_score": round(self.consistency_score, 3),
            "metadata": self.metadata,
        }


class SpeakerRepository:
    """
    In-memory speaker repository.

    For production, replace with SQLAlchemy-based implementation.
    This provides a simple starting point for local development.
    """

    def __init__(self):
        """Initialize the repository."""
        self._speakers: dict[str, Speaker] = {}
        self._by_employee_id: dict[str, str] = {}  # employee_id -> speaker_id

    async def create(
        self,
        employee_id: str,
        name: str,
        department: Optional[str] = None,
        embedding_count: int = 0,
        consistency_score: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> Speaker:
        """
        Create a new speaker.

        Args:
            employee_id: Unique employee identifier
            name: Speaker's display name
            department: Optional department
            embedding_count: Number of stored embeddings
            consistency_score: Enrollment consistency score
            metadata: Additional metadata

        Returns:
            Created Speaker object

        Raises:
            ValueError: If employee_id already exists
        """
        # Check for duplicate employee_id
        if employee_id in self._by_employee_id:
            raise ValueError(f"Employee ID already exists: {employee_id}")

        speaker_id = str(uuid.uuid4())
        now = datetime.utcnow()

        speaker = Speaker(
            id=speaker_id,
            employee_id=employee_id,
            name=name,
            department=department,
            enrolled_at=now,
            updated_at=now,
            is_active=True,
            embedding_count=embedding_count,
            consistency_score=consistency_score,
            metadata=metadata or {},
        )

        self._speakers[speaker_id] = speaker
        self._by_employee_id[employee_id] = speaker_id

        logger.info(
            "speaker_created",
            speaker_id=speaker_id,
            employee_id=employee_id,
            name=name,
        )

        return speaker

    async def get(self, speaker_id: str) -> Optional[Speaker]:
        """
        Get speaker by ID.

        Args:
            speaker_id: Speaker UUID

        Returns:
            Speaker or None if not found
        """
        return self._speakers.get(speaker_id)

    async def get_by_employee_id(self, employee_id: str) -> Optional[Speaker]:
        """
        Get speaker by employee ID.

        Args:
            employee_id: Employee identifier

        Returns:
            Speaker or None if not found
        """
        speaker_id = self._by_employee_id.get(employee_id)
        if speaker_id:
            return self._speakers.get(speaker_id)
        return None

    async def list(
        self,
        include_inactive: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Speaker]:
        """
        List speakers with pagination.

        Args:
            include_inactive: Whether to include inactive speakers
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of speakers
        """
        speakers = list(self._speakers.values())

        if not include_inactive:
            speakers = [s for s in speakers if s.is_active]

        # Sort by enrolled_at descending
        speakers.sort(key=lambda s: s.enrolled_at, reverse=True)

        return speakers[offset:offset + limit]

    async def update(
        self,
        speaker_id: str,
        name: Optional[str] = None,
        department: Optional[str] = None,
        is_active: Optional[bool] = None,
        embedding_count: Optional[int] = None,
        consistency_score: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[Speaker]:
        """
        Update speaker attributes.

        Args:
            speaker_id: Speaker UUID
            name: New name (optional)
            department: New department (optional)
            is_active: New active status (optional)
            embedding_count: New embedding count (optional)
            consistency_score: New consistency score (optional)
            metadata: Metadata to merge (optional)

        Returns:
            Updated Speaker or None if not found
        """
        speaker = self._speakers.get(speaker_id)
        if not speaker:
            return None

        if name is not None:
            speaker.name = name
        if department is not None:
            speaker.department = department
        if is_active is not None:
            speaker.is_active = is_active
        if embedding_count is not None:
            speaker.embedding_count = embedding_count
        if consistency_score is not None:
            speaker.consistency_score = consistency_score
        if metadata is not None:
            speaker.metadata.update(metadata)

        speaker.updated_at = datetime.utcnow()

        logger.info("speaker_updated", speaker_id=speaker_id)

        return speaker

    async def delete(
        self,
        speaker_id: str,
        hard_delete: bool = False,
    ) -> bool:
        """
        Delete a speaker.

        Args:
            speaker_id: Speaker UUID
            hard_delete: If True, permanently remove; otherwise soft-delete

        Returns:
            True if deleted, False if not found
        """
        speaker = self._speakers.get(speaker_id)
        if not speaker:
            return False

        if hard_delete:
            # Permanent deletion
            del self._speakers[speaker_id]
            if speaker.employee_id in self._by_employee_id:
                del self._by_employee_id[speaker.employee_id]
            logger.info("speaker_hard_deleted", speaker_id=speaker_id)
        else:
            # Soft delete
            speaker.is_active = False
            speaker.updated_at = datetime.utcnow()
            logger.info("speaker_soft_deleted", speaker_id=speaker_id)

        return True

    async def count(self, include_inactive: bool = False) -> int:
        """
        Count speakers.

        Args:
            include_inactive: Whether to count inactive speakers

        Returns:
            Number of speakers
        """
        if include_inactive:
            return len(self._speakers)
        return sum(1 for s in self._speakers.values() if s.is_active)

    async def exists(self, speaker_id: str) -> bool:
        """Check if speaker exists."""
        return speaker_id in self._speakers

    async def exists_by_employee_id(self, employee_id: str) -> bool:
        """Check if employee ID is already registered."""
        return employee_id in self._by_employee_id

    async def search(
        self,
        query: str,
        include_inactive: bool = False,
    ) -> List[Speaker]:
        """
        Search speakers by name or employee ID.

        Args:
            query: Search query
            include_inactive: Whether to include inactive speakers

        Returns:
            List of matching speakers
        """
        query_lower = query.lower()
        results = []

        for speaker in self._speakers.values():
            if not include_inactive and not speaker.is_active:
                continue

            if (
                query_lower in speaker.name.lower()
                or query_lower in speaker.employee_id.lower()
            ):
                results.append(speaker)

        return results
