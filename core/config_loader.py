"""
Config Loader - Loads and manages system configuration

Loads configuration from YAML files in the /config directory
and validates them using the config models.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import config models
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.config_models import (
    SystemConfig,
    ModelConfig,
    AgentConfig,
    ToolConfig,
    MemoryConfig,
    LoggingConfig,
    TaskQueueConfig,
    LogLevel,
    ModelProvider,
)


def _safe_enum(enum_cls, value, default):
    """Convert a string to an enum value, returning default on invalid input."""
    try:
        return enum_cls(value)
    except (ValueError, KeyError):
        print(
            f"[Config] Invalid value '{value}' for {enum_cls.__name__}, using '{default}'"
        )
        return default


class ConfigLoader:
    """
    Loads and manages system configuration from YAML files.

    Configuration files are stored in the /config directory:
    - settings.yaml: Main system settings
    - models.yaml: LLM model configuration
    - agents.yaml: Agent configurations
    - tools.yaml: Tool configurations
    - api_keys.yaml: API keys (should be in .gitignore)
    """

    # Project root is one level above this file (core/ -> project root)
    _PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    def __init__(self, config_dir: str = ""):
        """
        Initialize the config loader.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else ConfigLoader._PROJECT_ROOT / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Config file paths
        self.settings_path = self.config_dir / "settings.yaml"
        self.models_path = self.config_dir / "models.yaml"
        self.agents_path = self.config_dir / "agents.yaml"
        self.tools_path = self.config_dir / "tools.yaml"
        self.api_keys_path = self.config_dir / "api_keys.yaml"

        # Loaded configuration
        self.config: Optional[SystemConfig] = None

    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """
        Load a YAML file.

        Args:
            filepath: Path to the YAML file

        Returns:
            Dictionary of configuration data
        """
        if not filepath.exists():
            return {}

        with open(filepath, "r") as f:
            try:
                data = yaml.safe_load(f)
                return data if data is not None else {}
            except yaml.YAMLError as e:
                print(f"Error loading {filepath}: {e}")
                return {}

    def _save_yaml(self, filepath: Path, data: Dict[str, Any]):
        """
        Save data to a YAML file.

        Args:
            filepath: Path to save to
            data: Data to save
        """
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def load_model_config(self) -> ModelConfig:
        """Load model configuration"""
        data = self._load_yaml(self.models_path)

        if not data:
            # Return default if no config exists
            return ModelConfig()

        return ModelConfig(
            provider=ModelProvider(data.get("provider", "local")),
            model_name=data.get("model_name", "mistral-7b"),
            model_path=data.get("model_path"),
            api_key=data.get("api_key"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 2048),
            top_p=data.get("top_p", 0.9),
            context_window=data.get("context_window", 4096),
            use_gpu=data.get("use_gpu", True),
            gpu_layers=data.get("gpu_layers", 32),
            threads=data.get("threads", 4),
            batch_size=data.get("batch_size", 512),
        )

    def load_memory_config(self) -> MemoryConfig:
        """Load memory configuration"""
        data = self._load_yaml(self.settings_path)
        memory_data = data.get("memory", {})

        return MemoryConfig(
            vector_db_path=memory_data.get("vector_db_path", "./memory/vector_db"),
            embedding_model=memory_data.get("embedding_model", "all-MiniLM-L6-v2"),
            collection_name=memory_data.get("collection_name", "ai_memory"),
            short_term_days=memory_data.get("short_term_days", 7),
            long_term_threshold=memory_data.get("long_term_threshold", 0.7),
            max_memories=memory_data.get("max_memories", 10000),
            cleanup_interval=memory_data.get("cleanup_interval", 86400),
            max_search_results=memory_data.get("max_search_results", 10),
            similarity_threshold=memory_data.get("similarity_threshold", 0.6),
        )

    def load_logging_config(self) -> LoggingConfig:
        """Load logging configuration"""
        data = self._load_yaml(self.settings_path)
        log_data = data.get("logging", {})

        return LoggingConfig(
            log_level=_safe_enum(
                LogLevel, log_data.get("log_level", "info"), LogLevel.INFO
            ),
            log_dir=log_data.get("log_dir", "./logs"),
            log_file=log_data.get("log_file", "ai_system.log"),
            max_file_size=log_data.get("max_file_size", 10485760),
            backup_count=log_data.get("backup_count", 5),
            log_to_console=log_data.get("log_to_console", True),
            log_to_file=log_data.get("log_to_file", True),
        )

    def load_task_queue_config(self) -> TaskQueueConfig:
        """Load task queue configuration"""
        data = self._load_yaml(self.settings_path)
        queue_data = data.get("task_queue", {})

        return TaskQueueConfig(
            max_concurrent_tasks=queue_data.get("max_concurrent_tasks", 5),
            task_timeout=queue_data.get("task_timeout", 600),
            retry_delay=queue_data.get("retry_delay", 5),
            max_queue_size=queue_data.get("max_queue_size", 1000),
            priority_enabled=queue_data.get("priority_enabled", True),
        )

    def load_agents(self) -> List[AgentConfig]:
        """Load agent configurations"""
        data = self._load_yaml(self.agents_path)
        agents = []

        for agent_data in data.get("agents", []):
            agent = AgentConfig(
                agent_name=agent_data.get("name", ""),
                agent_type=agent_data.get("type", ""),
                enabled=agent_data.get("enabled", True),
                max_retries=agent_data.get("max_retries", 3),
                timeout=agent_data.get("timeout", 300),
                tools=agent_data.get("tools", []),
                permissions=agent_data.get("permissions", {}),
                settings=agent_data.get("settings", {}),
            )
            agents.append(agent)

        return agents

    def load_tools(self) -> List[ToolConfig]:
        """Load tool configurations"""
        data = self._load_yaml(self.tools_path)
        tools = []

        for tool_data in data.get("tools", []):
            tool = ToolConfig(
                tool_name=tool_data.get("name", ""),
                enabled=tool_data.get("enabled", True),
                requires_permission=tool_data.get("requires_permission", False),
                settings=tool_data.get("settings", {}),
            )
            tools.append(tool)

        return tools

    def load_api_keys(self) -> Dict[str, str]:
        """Load API keys, warning on any empty or missing values."""
        data = self._load_yaml(self.api_keys_path)
        keys = data.get("api_keys", {})
        for name, value in keys.items():
            if not value or not str(value).strip():
                print(f"[Config] Warning: API key '{name}' is empty or missing")
        return {k: str(v).strip() for k, v in keys.items() if v}

    def load_all(self) -> SystemConfig:
        """
        Load complete system configuration.

        Returns:
            SystemConfig object with all configuration loaded
        """
        # Load main settings
        settings_data = self._load_yaml(self.settings_path)

        # Create system config
        config = SystemConfig(
            system_name=settings_data.get("system_name", "AI System"),
            version=settings_data.get("version", "1.0.0"),
            debug_mode=settings_data.get("debug_mode", False),
            model=self.load_model_config(),
            memory=self.load_memory_config(),
            logging=self.load_logging_config(),
            task_queue=self.load_task_queue_config(),
            agents=self.load_agents(),
            tools=self.load_tools(),
            api_keys=self.load_api_keys(),
        )

        # Validate configuration
        errors = config.validate()
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")

        self.config = config
        return config

    def create_default_configs(self):
        """Create default configuration files if they don't exist"""

        # Default settings.yaml
        if not self.settings_path.exists():
            default_settings = {
                "system_name": "AI System",
                "version": "1.0.0",
                "debug_mode": False,
                "logging": {
                    "log_level": "info",
                    "log_dir": "./logs",
                    "log_file": "ai_system.log",
                    "max_file_size": 10485760,
                    "backup_count": 5,
                    "log_to_console": True,
                    "log_to_file": True,
                },
                "memory": {
                    "vector_db_path": "./memory/vector_db",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "collection_name": "ai_memory",
                    "short_term_days": 7,
                    "long_term_threshold": 0.7,
                    "max_memories": 10000,
                    "cleanup_interval": 86400,
                    "max_search_results": 10,
                    "similarity_threshold": 0.6,
                },
                "task_queue": {
                    "max_concurrent_tasks": 5,
                    "task_timeout": 600,
                    "retry_delay": 5,
                    "max_queue_size": 1000,
                    "priority_enabled": True,
                },
            }
            self._save_yaml(self.settings_path, default_settings)
            print("Created default settings.yaml")

        # Default models.yaml
        if not self.models_path.exists():
            default_models = {
                "provider": "local",
                "model_name": "mistral-7b",
                "model_path": "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.9,
                "context_window": 4096,
                "use_gpu": True,
                "gpu_layers": 32,
                "threads": 4,
                "batch_size": 512,
            }
            self._save_yaml(self.models_path, default_models)
            print("Created default models.yaml")

        # Default agents.yaml
        if not self.agents_path.exists():
            default_agents = {
                "agents": [
                    {
                        "name": "planning_agent",
                        "type": "planning",
                        "enabled": True,
                        "max_retries": 3,
                        "timeout": 300,
                        "tools": ["llm"],
                        "permissions": {"can_plan": True},
                    },
                    {
                        "name": "file_agent",
                        "type": "file",
                        "enabled": True,
                        "max_retries": 3,
                        "timeout": 300,
                        "tools": ["file_tools", "terminal"],
                        "permissions": {
                            "can_read": True,
                            "can_write": True,
                            "can_delete": False,
                        },
                    },
                    {
                        "name": "coding_agent",
                        "type": "coding",
                        "enabled": True,
                        "max_retries": 3,
                        "timeout": 600,
                        "tools": ["code_executor", "file_tools"],
                        "permissions": {"can_execute": True, "can_install": False},
                    },
                ]
            }
            self._save_yaml(self.agents_path, default_agents)
            print("Created default agents.yaml")

        # Default tools.yaml
        if not self.tools_path.exists():
            default_tools = {
                "tools": [
                    {
                        "name": "file_tools",
                        "enabled": True,
                        "requires_permission": False,
                    },
                    {
                        "name": "web_tools",
                        "enabled": True,
                        "requires_permission": False,
                    },
                    {
                        "name": "code_executor",
                        "enabled": True,
                        "requires_permission": True,
                    },
                    {"name": "terminal", "enabled": True, "requires_permission": True},
                    {
                        "name": "database_tools",
                        "enabled": True,
                        "requires_permission": False,
                    },
                ]
            }
            self._save_yaml(self.tools_path, default_tools)
            print("Created default tools.yaml")

        # Default api_keys.yaml (template only)
        if not self.api_keys_path.exists():
            default_api_keys = {
                "api_keys": {
                    "openai": "your-openai-key-here",
                    "anthropic": "your-anthropic-key-here",
                    "huggingface": "your-huggingface-key-here",
                }
            }
            self._save_yaml(self.api_keys_path, default_api_keys)
            print("Created default api_keys.yaml (remember to add to .gitignore!)")


# Global config instance
_config_instance: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Get the global configuration instance"""
    global _config_instance
    if _config_instance is None:
        loader = ConfigLoader()
        loader.create_default_configs()
        _config_instance = loader.load_all()
    return _config_instance


def reload_config(config_dir: str = "") -> SystemConfig:
    """
    Reload configuration from files.

    Args:
        config_dir: Directory containing config files

    Returns:
        Reloaded SystemConfig
    """
    global _config_instance
    loader = ConfigLoader(config_dir)
    _config_instance = loader.load_all()
    return _config_instance


if __name__ == "__main__":
    # Test the config loader
    print("Testing Config Loader...")

    # Create default configs
    loader = ConfigLoader(config_dir="./test_config")
    loader.create_default_configs()

    print("\n--- Loading Configuration ---")
    config = loader.load_all()

    print(f"\nSystem: {config.system_name} v{config.version}")
    print(f"Model: {config.model.model_name}")
    print(f"Log Level: {config.logging.log_level.value}")
    print(f"Memory DB Path: {config.memory.vector_db_path}")
    print(f"Max Concurrent Tasks: {config.task_queue.max_concurrent_tasks}")
    print(f"\nAgents loaded: {len(config.agents)}")
    for agent in config.agents:
        print(
            f"  - {agent.agent_name} ({agent.agent_type}) - {'Enabled' if agent.enabled else 'Disabled'}"
        )

    print(f"\nTools loaded: {len(config.tools)}")
    for tool in config.tools:
        print(f"  - {tool.tool_name} - {'Enabled' if tool.enabled else 'Disabled'}")

    print("\nConfig Loader test completed!")
