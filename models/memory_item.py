"""
Memory Item Model - Represents items stored in the memory system

Used by the Memory Manager to store and retrieve information
in the vector database and various memory stores.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(Enum):
    """Types of memory in the system"""

    SHORT_TERM = "short_term"  # Recent interactions
    LONG_TERM = "long_term"  # Persistent knowledge
    CONVERSATION = "conversation"  # Dialogue history
    KNOWLEDGE = "knowledge"  # Facts and information
    PROJECT = "project"  # Project-specific data
    FILE = "file"  # File-related memory
    AGENT = "agent"  # Agent-specific learnings
    TASK = "task"  # Task execution history
    REFLECTION = "reflection"  # System reflections and learnings


class MemoryImportance(Enum):
    """Importance level for memory retention"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryItem:
    """
    Represents a single item in the memory system.

    Used for vector DB storage, retrieval, and organization.
    """

    # Core identification
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Memory content
    content: str = ""  # The actual memory text
    embedding: Optional[List[float]] = None  # Vector embedding

    # Classification
    memory_type: MemoryType = MemoryType.SHORT_TERM
    importance: MemoryImportance = MemoryImportance.MEDIUM
    tags: List[str] = field(default_factory=list)

    # Context and relationships
    related_task_id: Optional[str] = None
    related_conversation_id: Optional[str] = None
    related_project: Optional[str] = None
    related_agent: Optional[str] = None
    related_memories: List[str] = field(default_factory=list)  # IDs of related memories

    # Metadata
    source: str = ""  # Where this memory came from
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    expires_at: Optional[datetime] = None  # For temporary memories

    # Relevance and quality
    relevance_score: float = 0.0  # 0.0 to 1.0
    confidence: float = 1.0  # How confident we are in this memory
    verified: bool = False  # Whether this has been verified/validated

    # Learning and reflection
    learned_from: Optional[str] = None  # What situation/task led to this learning
    success_indicator: Optional[bool] = None  # Was this associated with success?

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory item to dictionary for serialization"""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "embedding": self.embedding,
            "memory_type": self.memory_type.value,
            "importance": self.importance.value,
            "tags": self.tags,
            "related_task_id": self.related_task_id,
            "related_conversation_id": self.related_conversation_id,
            "related_project": self.related_project,
            "related_agent": self.related_agent,
            "related_memories": self.related_memories,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
            "access_count": self.access_count,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "relevance_score": self.relevance_score,
            "confidence": self.confidence,
            "verified": self.verified,
            "learned_from": self.learned_from,
            "success_indicator": self.success_indicator,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """Create MemoryItem from dictionary"""
        item = cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            content=data.get("content", ""),
            embedding=data.get("embedding"),
            memory_type=MemoryType(data.get("memory_type", "short_term")),
            importance=MemoryImportance(data.get("importance", 2)),
            tags=data.get("tags", []),
            related_task_id=data.get("related_task_id"),
            related_conversation_id=data.get("related_conversation_id"),
            related_project=data.get("related_project"),
            related_agent=data.get("related_agent"),
            related_memories=data.get("related_memories", []),
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            relevance_score=data.get("relevance_score", 0.0),
            confidence=data.get("confidence", 1.0),
            verified=data.get("verified", False),
            learned_from=data.get("learned_from"),
            success_indicator=data.get("success_indicator"),
        )

        # Handle datetime fields
        if data.get("created_at"):
            item.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_accessed"):
            item.last_accessed = datetime.fromisoformat(data["last_accessed"])
        if data.get("expires_at"):
            item.expires_at = datetime.fromisoformat(data["expires_at"])

        return item

    def access(self):
        """Record that this memory was accessed"""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def is_expired(self) -> bool:
        """Check if memory has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def should_retain(self) -> bool:
        """Determine if memory should be retained based on importance and access"""
        if self.importance == MemoryImportance.CRITICAL:
            return True
        if self.is_expired():
            return False
        if self.importance == MemoryImportance.LOW and self.access_count == 0:
            return False
        return True

    @classmethod
    def create_conversation_memory(
        cls, content: str, conversation_id: str, **kwargs
    ) -> "MemoryItem":
        """Convenience method to create conversation memory"""
        return cls(
            content=content,
            memory_type=MemoryType.CONVERSATION,
            related_conversation_id=conversation_id,
            source="conversation",
            **kwargs,
        )

    @classmethod
    def create_task_memory(
        cls, content: str, task_id: str, success: bool = True, **kwargs
    ) -> "MemoryItem":
        """Convenience method to create task memory"""
        return cls(
            content=content,
            memory_type=MemoryType.TASK,
            related_task_id=task_id,
            success_indicator=success,
            source="task_execution",
            **kwargs,
        )

    @classmethod
    def create_knowledge_memory(
        cls, content: str, verified: bool = False, **kwargs
    ) -> "MemoryItem":
        """Convenience method to create knowledge memory"""
        return cls(
            content=content,
            memory_type=MemoryType.KNOWLEDGE,
            importance=MemoryImportance.HIGH,
            verified=verified,
            source="learned_knowledge",
            **kwargs,
        )


if __name__ == "__main__":
    # Test the memory item model

    # Test conversation memory
    memory = MemoryItem.create_conversation_memory(
        content="User prefers Python over JavaScript",
        conversation_id="conv_123",
        tags=["preference", "programming"],
    )
    print(f"Created memory: {memory.memory_id}")
    print(f"Type: {memory.memory_type.value}")
    print(f"Content: {memory.content}")

    # Test access tracking
    memory.access()
    memory.access()
    print(f"\nAccess count: {memory.access_count}")
    print(f"Last accessed: {memory.last_accessed}")

    # Test serialization
    memory_dict = memory.to_dict()
    print(f"\nSerialized: {memory_dict['memory_type']}")

    # Test deserialization
    memory2 = MemoryItem.from_dict(memory_dict)
    print(f"\nDeserialized: {memory2.memory_id}")
    print(f"Match: {memory.memory_id == memory2.memory_id}")

    # Test retention logic
    print(f"\nShould retain: {memory.should_retain()}")
