"""
Logger System - Centralized logging for the AI system

Logs everything: user requests, plans, tasks, agent actions, tool usage,
errors, results, reflections, and system events.

Logs are stored in /logs directory with rotation support.
"""

import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


class AILogger:
    """
    Centralized logger for the AI system.

    Provides structured logging with automatic rotation,
    multiple output targets, and event categorization.
    """

    # Project root is one level above this file (core/ -> project root)
    _PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    def __init__(
        self,
        log_dir: str = "",
        log_file: str = "ai_system.log",
        log_level: str = "INFO",
        max_file_size: int = 10485760,  # 10 MB
        backup_count: int = 5,
        log_to_console: bool = True,
        log_to_file: bool = True,
    ):
        """
        Initialize the logger.

        Args:
            log_dir: Directory to store log files
            log_file: Main log file name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_file_size: Maximum size of log file before rotation (bytes)
            backup_count: Number of backup files to keep
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
        """
        self.log_dir = Path(log_dir) if log_dir else AILogger._PROJECT_ROOT / "logs"
        self.log_file = log_file
        _valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        _normalized = log_level.upper() if log_level else "INFO"
        if _normalized not in _valid:
            print(f"[Logger] Invalid log_level '{log_level}', defaulting to INFO")
            _normalized = "INFO"
        self.log_level = getattr(logging, _normalized)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = logging.getLogger("AISystem")
        self.logger.setLevel(self.log_level)

        # Remove any existing handlers
        self.logger.handlers.clear()

        # Setup handlers
        self._setup_handlers()

        # Create specialized log files for different event types
        self._setup_specialized_loggers()

    def _setup_handlers(self):
        """Setup logging handlers for console and file output"""

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        if self.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler with rotation
        if self.log_to_file:
            log_path = self.log_dir / self.log_file
            file_handler = RotatingFileHandler(
                log_path, maxBytes=self.max_file_size, backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _setup_specialized_loggers(self):
        """Create specialized log files for different event types"""
        self.event_log_path = self.log_dir / "events.jsonl"
        self.task_log_path = self.log_dir / "tasks.jsonl"
        self.agent_log_path = self.log_dir / "agents.jsonl"
        self.error_log_path = self.log_dir / "errors.jsonl"
        self.reflection_log_path = self.log_dir / "reflections.jsonl"

    def _write_json_log(self, filepath: Path, data: Dict[str, Any]):
        """Write a JSON line to a specialized log file"""
        log_entry = {"timestamp": datetime.now().isoformat(), **data}
        with open(filepath, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # Standard logging methods

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
        # Also write to error log
        self._write_json_log(
            self.error_log_path, {"level": "error", "message": message, **kwargs}
        )

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)
        # Also write to error log
        self._write_json_log(
            self.error_log_path, {"level": "critical", "message": message, **kwargs}
        )

    # Specialized logging methods

    def log_event(self, event_type: str, description: str, **kwargs):
        """
        Log a system event.

        Args:
            event_type: Type of event (e.g., "startup", "shutdown", "config_change")
            description: Description of the event
            **kwargs: Additional event data
        """
        self.info(f"[EVENT:{event_type}] {description}")
        self._write_json_log(
            self.event_log_path,
            {"event_type": event_type, "description": description, **kwargs},
        )

    def log_user_request(self, request: str, user_id: Optional[str] = None, **kwargs):
        """Log a user request"""
        self.info(f"[USER REQUEST] {request}")
        self._write_json_log(
            self.event_log_path,
            {
                "event_type": "user_request",
                "request": request,
                "user_id": user_id,
                **kwargs,
            },
        )

    def log_task(self, task_id: str, action: str, details: Dict[str, Any]):
        """
        Log task-related events.

        Args:
            task_id: Task identifier
            action: Action taken (e.g., "created", "started", "completed", "failed")
            details: Additional task details
        """
        self.info(f"[TASK:{task_id}] {action}")
        self._write_json_log(
            self.task_log_path, {"task_id": task_id, "action": action, **details}
        )

    def log_agent_action(self, agent_name: str, action: str, details: Dict[str, Any]):
        """
        Log agent actions.

        Args:
            agent_name: Name of the agent
            action: Action performed
            details: Additional details
        """
        self.info(f"[AGENT:{agent_name}] {action}")
        self._write_json_log(
            self.agent_log_path, {"agent_name": agent_name, "action": action, **details}
        )

    def log_tool_usage(self, tool_name: str, agent: str, details: Dict[str, Any]):
        """
        Log tool usage by agents.

        Args:
            tool_name: Name of the tool
            agent: Agent using the tool
            details: Usage details
        """
        self.info(f"[TOOL:{tool_name}] Used by {agent}")
        self._write_json_log(
            self.agent_log_path,
            {
                "event_type": "tool_usage",
                "tool_name": tool_name,
                "agent": agent,
                **details,
            },
        )

    def log_reflection(
        self, reflection_type: str, content: str, metadata: Dict[str, Any]
    ):
        """
        Log system reflections and learnings.

        Args:
            reflection_type: Type of reflection (e.g., "task_evaluation", "improvement")
            content: Reflection content
            metadata: Additional metadata
        """
        self.info(f"[REFLECTION:{reflection_type}] {content[:100]}...")
        self._write_json_log(
            self.reflection_log_path,
            {"reflection_type": reflection_type, "content": content, **metadata},
        )

    def log_plan(self, plan_id: str, steps: list, details: Dict[str, Any]):
        """
        Log execution plans created by the planner.

        Args:
            plan_id: Plan identifier
            steps: List of plan steps
            details: Additional plan details
        """
        self.info(f"[PLAN:{plan_id}] Created with {len(steps)} steps")
        self._write_json_log(
            self.task_log_path,
            {
                "event_type": "plan_created",
                "plan_id": plan_id,
                "steps": steps,
                **details,
            },
        )

    def log_performance(self, component: str, metrics: Dict[str, Any]):
        """
        Log performance metrics.

        Args:
            component: Component being measured
            metrics: Performance metrics
        """
        self.info(f"[PERFORMANCE:{component}] {metrics}")
        self._write_json_log(
            self.event_log_path,
            {"event_type": "performance", "component": component, **metrics},
        )

    def get_recent_logs(self, log_type: str = "main", lines: int = 100) -> list:
        """
        Get recent log entries.

        Args:
            log_type: Type of log ("main", "events", "tasks", "agents", "errors", "reflections")
            lines: Number of recent lines to retrieve

        Returns:
            List of recent log entries
        """
        log_map = {
            "main": self.log_dir / self.log_file,
            "events": self.event_log_path,
            "tasks": self.task_log_path,
            "agents": self.agent_log_path,
            "errors": self.error_log_path,
            "reflections": self.reflection_log_path,
        }

        log_path = log_map.get(log_type, self.log_dir / self.log_file)

        if not log_path.exists():
            return []

        with open(log_path, "r") as f:
            all_lines = f.readlines()
            return all_lines[-lines:]


# Global logger instance
_logger_instance: Optional[AILogger] = None


def get_logger() -> AILogger:
    """Get the global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AILogger()
    return _logger_instance


def init_logger(
    log_dir: str = "",
    log_file: str = "ai_system.log",
    log_level: str = "INFO",
    **kwargs,
) -> AILogger:
    """
    Initialize the global logger.

    Args:
        log_dir: Directory for log files
        log_file: Main log file name
        log_level: Logging level
        **kwargs: Additional logger configuration

    Returns:
        Initialized AILogger instance
    """
    global _logger_instance
    _logger_instance = AILogger(
        log_dir=log_dir, log_file=log_file, log_level=log_level, **kwargs
    )
    return _logger_instance


if __name__ == "__main__":
    # Test the logger
    logger = init_logger(log_dir="./test_logs", log_level="DEBUG")

    # Test standard logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")

    # Test specialized logging
    logger.log_event("test_startup", "System started in test mode")
    logger.log_user_request("Test user request", user_id="user_123")
    logger.log_task("task_001", "created", {"description": "Test task"})
    logger.log_agent_action("TestAgent", "executed", {"result": "success"})
    logger.log_tool_usage("file_reader", "FileAgent", {"file": "test.txt"})
    logger.log_reflection(
        "test_reflection", "This is a test reflection", {"confidence": 0.9}
    )
    logger.log_plan("plan_001", ["step1", "step2"], {"priority": "high"})

    print("\nLogger test completed. Check ./test_logs directory for output files.")

    # Test reading recent logs
    recent = logger.get_recent_logs("events", lines=5)
    print(f"\nRecent event logs ({len(recent)} entries):")
    for entry in recent:
        print(entry.strip())
