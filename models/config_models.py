"""
Config Models - Data structures for system configuration

Used by the Config Loader to validate and manage system settings.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LogLevel(Enum):
    """Logging levels"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ModelProvider(Enum):
    """LLM providers"""

    LOCAL = "local"  # Local models
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


@dataclass
class ModelConfig:
    """Configuration for LLM models"""

    provider: ModelProvider = ModelProvider.LOCAL
    model_name: str = "mistral-7b"
    model_path: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    context_window: int = 4096

    # Performance settings
    use_gpu: bool = True
    gpu_layers: int = 32
    threads: int = 4
    batch_size: int = 512

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "context_window": self.context_window,
            "use_gpu": self.use_gpu,
            "gpu_layers": self.gpu_layers,
            "threads": self.threads,
            "batch_size": self.batch_size,
        }


@dataclass
class AgentConfig:
    """Configuration for agents"""

    agent_name: str
    agent_type: str
    enabled: bool = True
    max_retries: int = 3
    timeout: int = 300  # seconds
    tools: List[str] = field(default_factory=list)
    permissions: Dict[str, bool] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "enabled": self.enabled,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "tools": self.tools,
            "permissions": self.permissions,
            "settings": self.settings,
        }


@dataclass
class ToolConfig:
    """Configuration for tools"""

    tool_name: str
    enabled: bool = True
    requires_permission: bool = False
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tool_name": self.tool_name,
            "enabled": self.enabled,
            "requires_permission": self.requires_permission,
            "settings": self.settings,
        }


@dataclass
class MemoryConfig:
    """Configuration for memory system"""

    vector_db_path: str = "./memory/vector_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "ai_memory"

    # Memory retention
    short_term_days: int = 7
    long_term_threshold: float = 0.7  # Importance threshold for long-term storage
    max_memories: int = 10000
    cleanup_interval: int = 86400  # seconds (1 day)

    # Search settings
    max_search_results: int = 10
    similarity_threshold: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "vector_db_path": self.vector_db_path,
            "embedding_model": self.embedding_model,
            "collection_name": self.collection_name,
            "short_term_days": self.short_term_days,
            "long_term_threshold": self.long_term_threshold,
            "max_memories": self.max_memories,
            "cleanup_interval": self.cleanup_interval,
            "max_search_results": self.max_search_results,
            "similarity_threshold": self.similarity_threshold,
        }


@dataclass
class LoggingConfig:
    """Configuration for logging"""

    log_level: LogLevel = LogLevel.INFO
    log_dir: str = "./logs"
    log_file: str = "ai_system.log"
    max_file_size: int = 10485760  # 10 MB
    backup_count: int = 5
    log_to_console: bool = True
    log_to_file: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "log_level": self.log_level.value,
            "log_dir": self.log_dir,
            "log_file": self.log_file,
            "max_file_size": self.max_file_size,
            "backup_count": self.backup_count,
            "log_to_console": self.log_to_console,
            "log_to_file": self.log_to_file,
        }


@dataclass
class TaskQueueConfig:
    """Configuration for task queue"""

    max_concurrent_tasks: int = 5
    task_timeout: int = 600  # seconds
    retry_delay: int = 5  # seconds
    max_queue_size: int = 1000
    priority_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "task_timeout": self.task_timeout,
            "retry_delay": self.retry_delay,
            "max_queue_size": self.max_queue_size,
            "priority_enabled": self.priority_enabled,
        }


@dataclass
class SystemConfig:
    """Main system configuration"""

    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    task_queue: TaskQueueConfig = field(default_factory=TaskQueueConfig)

    # System settings
    system_name: str = "AI System"
    version: str = "1.0.0"
    debug_mode: bool = False

    # Agents and tools (loaded separately)
    agents: List[AgentConfig] = field(default_factory=list)
    tools: List[ToolConfig] = field(default_factory=list)

    # API keys (loaded from separate secure file)
    api_keys: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary"""
        return {
            "system_name": self.system_name,
            "version": self.version,
            "debug_mode": self.debug_mode,
            "model": self.model.to_dict(),
            "memory": self.memory.to_dict(),
            "logging": self.logging.to_dict(),
            "task_queue": self.task_queue.to_dict(),
            "agents": [agent.to_dict() for agent in self.agents],
            "tools": [tool.to_dict() for tool in self.tools],
        }

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Validate model config
        if self.model.max_tokens <= 0:
            errors.append("model.max_tokens must be positive")
        if self.model.temperature < 0 or self.model.temperature > 2:
            errors.append("model.temperature must be between 0 and 2")

        # Validate memory config
        if self.memory.max_memories <= 0:
            errors.append("memory.max_memories must be positive")
        if self.memory.similarity_threshold < 0 or self.memory.similarity_threshold > 1:
            errors.append("memory.similarity_threshold must be between 0 and 1")

        # Validate task queue
        if self.task_queue.max_concurrent_tasks <= 0:
            errors.append("task_queue.max_concurrent_tasks must be positive")

        return errors


if __name__ == "__main__":
    # Test configuration models

    # Create default config
    config = SystemConfig()
    print(f"System: {config.system_name} v{config.version}")
    print(f"Model: {config.model.model_name}")
    print(f"Log level: {config.logging.log_level.value}")

    # Validate
    errors = config.validate()
    print(f"\nValidation errors: {len(errors)}")

    # Test custom config
    config2 = SystemConfig(
        model=ModelConfig(model_name="mistral-7b-instruct", temperature=0.8),
        memory=MemoryConfig(max_memories=5000),
    )
    print(f"\nCustom model: {config2.model.model_name}")
    print(f"Max memories: {config2.memory.max_memories}")

    # Test serialization
    config_dict = config2.to_dict()
    print(f"\nSerialized config keys: {list(config_dict.keys())}")
