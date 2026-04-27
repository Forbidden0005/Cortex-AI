"""
Agent Result Model - Standardized format for agent responses

All agents must return results in this format for consistency.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ResultStatus(Enum):
    """Status of agent result"""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # Partially completed
    ERROR = "error"


@dataclass
class AgentResult:
    """
    Standardized result format returned by all agents.

    This ensures the orchestrator and other components can
    consistently process agent outputs.
    """

    # Core result data
    status: ResultStatus
    data: Any = None  # The actual result data

    # Agent identification
    agent_name: str = ""
    agent_type: str = ""

    # Task context
    task_id: str = ""
    task_description: str = ""

    # Result metadata
    message: str = ""  # Human-readable description of what happened
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Performance metrics
    execution_time: float = 0.0  # seconds
    tokens_used: Optional[int] = None  # For LLM operations
    cost: Optional[float] = None  # Estimated cost

    # Tools and resources used
    tools_used: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_read: List[str] = field(default_factory=list)

    # Follow-up actions
    suggested_next_steps: List[str] = field(default_factory=list)
    requires_user_input: bool = False
    user_prompt: Optional[str] = None

    # Memory and learning
    save_to_memory: bool = True
    memory_tags: List[str] = field(default_factory=list)
    learnings: List[str] = field(default_factory=list)  # Insights gained

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "status": self.status.value,
            "data": self.data,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "message": self.message,
            "error": self.error,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "tools_used": self.tools_used,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "files_read": self.files_read,
            "suggested_next_steps": self.suggested_next_steps,
            "requires_user_input": self.requires_user_input,
            "user_prompt": self.user_prompt,
            "save_to_memory": self.save_to_memory,
            "memory_tags": self.memory_tags,
            "learnings": self.learnings,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResult":
        """Create AgentResult from dictionary"""
        result = cls(
            status=ResultStatus(data.get("status", "success")),
            data=data.get("data"),
            agent_name=data.get("agent_name", ""),
            agent_type=data.get("agent_type", ""),
            task_id=data.get("task_id", ""),
            task_description=data.get("task_description", ""),
            message=data.get("message", ""),
            error=data.get("error"),
            warnings=data.get("warnings", []),
            execution_time=data.get("execution_time", 0.0),
            tokens_used=data.get("tokens_used"),
            cost=data.get("cost"),
            tools_used=data.get("tools_used", []),
            files_created=data.get("files_created", []),
            files_modified=data.get("files_modified", []),
            files_read=data.get("files_read", []),
            suggested_next_steps=data.get("suggested_next_steps", []),
            requires_user_input=data.get("requires_user_input", False),
            user_prompt=data.get("user_prompt"),
            save_to_memory=data.get("save_to_memory", True),
            memory_tags=data.get("memory_tags", []),
            learnings=data.get("learnings", []),
            metadata=data.get("metadata", {}),
        )

        if data.get("timestamp"):
            result.timestamp = datetime.fromisoformat(data["timestamp"])

        return result

    def is_successful(self) -> bool:
        """Check if the result represents success"""
        return self.status in [ResultStatus.SUCCESS, ResultStatus.PARTIAL]

    def has_errors(self) -> bool:
        """Check if the result has errors"""
        return (
            self.status in [ResultStatus.ERROR, ResultStatus.FAILURE]
            or self.error is not None
        )

    @classmethod
    def success(
        cls, data: Any, agent_name: str = "", message: str = "", **kwargs
    ) -> "AgentResult":
        """Convenience method to create a success result"""
        return cls(
            status=ResultStatus.SUCCESS,
            data=data,
            agent_name=agent_name,
            message=message,
            **kwargs,
        )

    @classmethod
    def failure(
        cls, error: str, agent_name: str = "", message: str = "", **kwargs
    ) -> "AgentResult":
        """Convenience method to create a failure result"""
        return cls(
            status=ResultStatus.FAILURE,
            error=error,
            agent_name=agent_name,
            message=message,
            **kwargs,
        )

    @classmethod
    def create_error(cls, error: str, agent_name: str = "", **kwargs) -> "AgentResult":
        """Convenience method to create an error result"""
        return cls(
            status=ResultStatus.ERROR,
            error=error,
            agent_name=agent_name,
            message=f"Error: {error}",
            **kwargs,
        )


if __name__ == "__main__":
    # Test the agent result model

    # Test success result
    result = AgentResult.success(
        data={"output": "test.txt", "lines": 100},
        agent_name="FileAgent",
        message="File created successfully",
        files_created=["test.txt"],
    )
    print(f"Success result: {result.status.value}")
    print(f"Data: {result.data}")

    # Test failure result
    result2 = AgentResult.failure(
        error="Permission denied",
        agent_name="FileAgent",
        message="Could not create file",
    )
    print(f"\nFailure result: {result2.status.value}")
    print(f"Error: {result2.error}")

    # Test serialization
    result_dict = result.to_dict()
    print(f"\nSerialized: {result_dict}")

    # Test deserialization
    result3 = AgentResult.from_dict(result_dict)
    print(f"\nDeserialized: {result3.agent_name}")
    print(f"Match: {result.agent_name == result3.agent_name}")
