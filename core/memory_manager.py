"""
Memory Manager - Handles all memory operations for the AI system

Supports multiple backends:
- ChromaDB (vector database) - preferred
- File-based (JSON fallback) - when ChromaDB unavailable

Memory types: short-term, long-term, conversation, knowledge, project, file, agent, task, reflection
"""

import json
# Import models
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))
from models import MemoryImportance, MemoryItem, MemoryType

# Try to import ChromaDB
try:
    import chromadb

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class MemoryBackend:
    """Base class for memory backends"""

    def save_memory(self, memory: MemoryItem) -> bool:
        raise NotImplementedError

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        raise NotImplementedError

    def search_memories(
        self, query: str, limit: int = 10, memory_type: Optional[MemoryType] = None
    ) -> List[MemoryItem]:
        raise NotImplementedError

    def delete_memory(self, memory_id: str) -> bool:
        raise NotImplementedError

    def get_all_memories(
        self, memory_type: Optional[MemoryType] = None
    ) -> List[MemoryItem]:
        raise NotImplementedError


class FileBasedBackend(MemoryBackend):
    """Simple file-based memory backend using JSON"""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each memory type
        for mem_type in MemoryType:
            type_dir = self.storage_dir / mem_type.value
            type_dir.mkdir(exist_ok=True)

        # Index file for fast lookups
        self.index_file = self.storage_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        """Load memory index (memory_id -> filepath)"""
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save memory index"""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f)

    def _get_filepath(self, memory: MemoryItem) -> Path:
        """Get filepath for a memory"""
        type_dir = self.storage_dir / memory.memory_type.value
        return type_dir / f"{memory.memory_id}.json"

    def save_memory(self, memory: MemoryItem) -> bool:
        """Save memory to file"""
        try:
            filepath = self._get_filepath(memory)
            with open(filepath, "w") as f:
                json.dump(memory.to_dict(), f, indent=2)

            # Update index
            self.index[memory.memory_id] = str(filepath)
            self._save_index()

            return True
        except Exception as e:
            print(f"Error saving memory: {e}")
            return False

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Get memory by ID"""
        filepath = self.index.get(memory_id)
        if not filepath or not Path(filepath).exists():
            return None

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return MemoryItem.from_dict(data)
        except Exception as e:
            print(f"Error loading memory: {e}")
            return None

    def search_memories(
        self, query: str, limit: int = 10, memory_type: Optional[MemoryType] = None
    ) -> List[MemoryItem]:
        """Search memories by text (simple keyword matching)"""
        results = []
        query_lower = query.lower()

        for memory_id in self.index:
            memory = self.get_memory(memory_id)
            if not memory:
                continue

            # Filter by type if specified
            if memory_type and memory.memory_type != memory_type:
                continue

            # Simple keyword matching
            if query_lower in memory.content.lower():
                words = memory.content.split()
                memory.relevance_score = (
                    memory.content.lower().count(query_lower) / len(words)
                    if words
                    else 0.0
                )
                results.append(memory)

        # Sort by relevance
        results.sort(key=lambda m: m.relevance_score, reverse=True)
        return results[:limit]

    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory"""
        filepath = self.index.get(memory_id)
        if not filepath:
            return False

        try:
            Path(filepath).unlink()
            del self.index[memory_id]
            self._save_index()
            return True
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False

    def get_all_memories(
        self, memory_type: Optional[MemoryType] = None
    ) -> List[MemoryItem]:
        """Get all memories, optionally filtered by type"""
        memories = []
        for memory_id in self.index:
            memory = self.get_memory(memory_id)
            if memory:
                if memory_type is None or memory.memory_type == memory_type:
                    memories.append(memory)
        return memories


class ChromaDBBackend(MemoryBackend):
    """ChromaDB vector database backend"""

    def __init__(self, db_path: str, collection_name: str = "ai_memory"):
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB not available. Install with: pip install chromadb"
            )

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def save_memory(self, memory: MemoryItem) -> bool:
        """Save memory to ChromaDB"""
        try:
            self.collection.add(
                ids=[memory.memory_id],
                documents=[memory.content],
                metadatas=[
                    {
                        "memory_type": memory.memory_type.value,
                        "importance": memory.importance.value,
                        "tags": ",".join(memory.tags),
                        "created_at": memory.created_at.isoformat(),
                        "source": memory.source,
                    }
                ],
            )
            return True
        except Exception as e:
            print(f"Error saving to ChromaDB: {e}")
            return False

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Get memory by ID"""
        try:
            results = self.collection.get(ids=[memory_id])
            if not results["ids"]:
                return None

            # Reconstruct memory item
            return MemoryItem(
                memory_id=results["ids"][0],
                content=results["documents"][0],
                memory_type=MemoryType(results["metadatas"][0]["memory_type"]),
                importance=MemoryImportance(results["metadatas"][0]["importance"]),
                tags=(
                    results["metadatas"][0]["tags"].split(",")
                    if results["metadatas"][0]["tags"]
                    else []
                ),
                source=results["metadatas"][0]["source"],
            )
        except Exception as e:
            print(f"Error getting memory: {e}")
            return None

    def search_memories(
        self, query: str, limit: int = 10, memory_type: Optional[MemoryType] = None
    ) -> List[MemoryItem]:
        """Search memories using vector similarity"""
        try:
            where_filter = None
            if memory_type:
                where_filter = {"memory_type": memory_type.value}

            results = self.collection.query(
                query_texts=[query], n_results=limit, where=where_filter
            )

            memories = []
            for i, memory_id in enumerate(results["ids"][0]):
                memory = MemoryItem(
                    memory_id=memory_id,
                    content=results["documents"][0][i],
                    memory_type=MemoryType(results["metadatas"][0][i]["memory_type"]),
                    importance=MemoryImportance(
                        results["metadatas"][0][i]["importance"]
                    ),
                    relevance_score=1.0
                    - results["distances"][0][i],  # Convert distance to similarity
                )
                memories.append(memory)

            return memories
        except Exception as e:
            print(f"Error searching memories: {e}")
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory"""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False

    def get_all_memories(
        self, memory_type: Optional[MemoryType] = None
    ) -> List[MemoryItem]:
        """Get all memories"""
        try:
            where_filter = {"memory_type": memory_type.value} if memory_type else None
            results = self.collection.get(where=where_filter)

            memories = []
            for i, memory_id in enumerate(results["ids"]):
                memory = MemoryItem(
                    memory_id=memory_id,
                    content=results["documents"][i],
                    memory_type=MemoryType(results["metadatas"][i]["memory_type"]),
                    importance=MemoryImportance(results["metadatas"][i]["importance"]),
                )
                memories.append(memory)

            return memories
        except Exception as e:
            print(f"Error getting all memories: {e}")
            return []


class MemoryManager:
    """
    Central memory management system.

    Handles all memory operations including:
    - Saving memories
    - Retrieving memories
    - Searching memories
    - Organizing by type
    - Cleaning old memories
    """

    # Project root is one level above this file (core/ -> project root)
    _PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    def __init__(
        self,
        storage_dir: str = "",
        use_chromadb: bool = False,
        collection_name: str = "ai_memory",
    ):
        """
        Initialize memory manager.

        Args:
            storage_dir: Directory for memory storage
            use_chromadb: Whether to use ChromaDB (requires installation)
            collection_name: ChromaDB collection name
        """
        if not storage_dir:
            self.storage_dir = MemoryManager._PROJECT_ROOT / "memory"
        else:
            _p = Path(storage_dir)
            self.storage_dir = _p if _p.is_absolute() else MemoryManager._PROJECT_ROOT / _p
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize backend
        if use_chromadb and CHROMADB_AVAILABLE:
            print("Using ChromaDB backend")
            self.backend = ChromaDBBackend(
                str(self.storage_dir / "vector_db"), collection_name
            )
        else:
            if use_chromadb and not CHROMADB_AVAILABLE:
                print("ChromaDB not available, falling back to file-based storage")
            else:
                print("Using file-based backend")
            self.backend = FileBasedBackend(str(self.storage_dir))

        # Memory type directories
        self.type_dirs = {
            MemoryType.SHORT_TERM: self.storage_dir / "short_term",
            MemoryType.LONG_TERM: self.storage_dir / "long_term",
            MemoryType.CONVERSATION: self.storage_dir / "conversations",
            MemoryType.KNOWLEDGE: self.storage_dir / "knowledge",
            MemoryType.PROJECT: self.storage_dir / "projects",
            MemoryType.FILE: self.storage_dir / "files",
            MemoryType.AGENT: self.storage_dir / "agent_memory",
            MemoryType.TASK: self.storage_dir / "tasks",
            MemoryType.REFLECTION: self.storage_dir / "reflections",
        }

        # Create directories
        for dir_path in self.type_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def save(self, memory: MemoryItem) -> bool:
        """
        Save a memory.

        Args:
            memory: MemoryItem to save

        Returns:
            True if successful
        """
        return self.backend.save_memory(memory)

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Get a memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            MemoryItem if found, None otherwise
        """
        memory = self.backend.get_memory(memory_id)
        if memory:
            # access() always increments access_count; persist the updated state
            memory.access()
            self.backend.save_memory(memory)
        return memory

    def search(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        min_importance: Optional[MemoryImportance] = None,
    ) -> List[MemoryItem]:
        """
        Search memories.

        Args:
            query: Search query
            limit: Maximum number of results
            memory_type: Filter by memory type
            min_importance: Minimum importance level

        Returns:
            List of matching memories
        """
        results = self.backend.search_memories(query, limit * 2, memory_type)

        # Filter by importance if specified
        if min_importance:
            results = [m for m in results if m.importance.value >= min_importance.value]

        # Update access
        for memory in results[:limit]:
            memory.access()
            self.backend.save_memory(memory)

        return results[:limit]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory"""
        return self.backend.delete_memory(memory_id)

    def get_by_type(self, memory_type: MemoryType) -> List[MemoryItem]:
        """Get all memories of a specific type"""
        return self.backend.get_all_memories(memory_type)

    def get_recent(
        self, memory_type: Optional[MemoryType] = None, limit: int = 10
    ) -> List[MemoryItem]:
        """Get recent memories"""
        memories = self.backend.get_all_memories(memory_type)
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    def cleanup_old_memories(self, days: int = 30):
        """
        Clean up old, unimportant memories.

        Args:
            days: Delete memories older than this many days (if low importance)
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        all_memories = self.backend.get_all_memories()

        deleted = 0
        for memory in all_memories:
            # Don't delete critical or high importance memories
            if memory.importance in [MemoryImportance.CRITICAL, MemoryImportance.HIGH]:
                continue

            # Don't delete recently accessed memories
            if memory.last_accessed and memory.last_accessed > cutoff_date:
                continue

            # Delete if old and not accessed
            if memory.created_at < cutoff_date and memory.access_count == 0:
                if self.backend.delete_memory(memory.memory_id):
                    deleted += 1

        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        all_memories = self.backend.get_all_memories()

        stats = {
            "total": len(all_memories),
            "by_type": {},
            "by_importance": {},
            "total_accesses": sum(m.access_count for m in all_memories),
        }

        # Count by type
        for mem_type in MemoryType:
            count = len([m for m in all_memories if m.memory_type == mem_type])
            stats["by_type"][mem_type.value] = count

        # Count by importance
        for importance in MemoryImportance:
            count = len([m for m in all_memories if m.importance == importance])
            stats["by_importance"][importance.value] = count

        return stats


if __name__ == "__main__":
    # Test the memory manager
    print("Testing Memory Manager...")

    # Initialize with file-based backend
    manager = MemoryManager(storage_dir="./test_memory", use_chromadb=False)

    # Create and save memories
    print("\n--- Creating Memories ---")

    memory1 = MemoryItem.create_conversation_memory(
        content="User prefers Python over JavaScript for scripting",
        conversation_id="conv_001",
        tags=["preference", "programming", "python"],
    )
    memory1.importance = MemoryImportance.HIGH
    manager.save(memory1)
    print(f"Saved conversation memory: {memory1.memory_id}")

    memory2 = MemoryItem.create_task_memory(
        content="Successfully generated PDF report with charts",
        task_id="task_001",
        success=True,
        tags=["pdf", "success", "charts"],
    )
    memory2.importance = MemoryImportance.MEDIUM
    manager.save(memory2)
    print(f"Saved task memory: {memory2.memory_id}")

    memory3 = MemoryItem.create_knowledge_memory(
        content="Python's asyncio requires event loop for concurrent operations",
        verified=True,
        tags=["python", "asyncio", "knowledge"],
    )
    memory3.importance = MemoryImportance.HIGH
    manager.save(memory3)
    print(f"Saved knowledge memory: {memory3.memory_id}")

    # Search memories
    print("\n--- Searching Memories ---")
    results = manager.search("python", limit=5)
    print(f"Found {len(results)} memories for 'python':")
    for mem in results:
        print(f"  - {mem.content[:60]}... (score: {mem.relevance_score:.2f})")

    # Get by type
    print("\n--- Memories by Type ---")
    conv_memories = manager.get_by_type(MemoryType.CONVERSATION)
    print(f"Conversation memories: {len(conv_memories)}")

    task_memories = manager.get_by_type(MemoryType.TASK)
    print(f"Task memories: {len(task_memories)}")

    # Get recent
    print("\n--- Recent Memories ---")
    recent = manager.get_recent(limit=3)
    for mem in recent:
        print(f"  - [{mem.memory_type.value}] {mem.content[:50]}...")

    # Statistics
    print("\n--- Memory Statistics ---")
    stats = manager.get_stats()
    print(f"Total memories: {stats['total']}")
    print(f"Total accesses: {stats['total_accesses']}")
    print(f"By type: {stats['by_type']}")
    print(f"By importance: {stats['by_importance']}")

    print("\nMemory Manager test completed!")
