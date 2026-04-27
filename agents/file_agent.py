import os

from agents.base_agent import BaseAgent
from models import Task


class FileAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="FileAgent", agent_type="file", description="Handles file operations"
        )

    def _do_work(self, task: "Task") -> dict:  # noqa: F821
        params = task.parameters or {}
        operation = params.get("operation", "read")
        filepath = params.get("filepath")

        # Validate required parameter early
        if operation in ("read", "write", "list") and not filepath:
            raise ValueError(
                f"FileAgent requires 'filepath' in task.parameters for "
                f"operation '{operation}'"
            )
        if operation == "read":
            self._track_file_read(filepath)
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            # Warn if replacement characters were needed (indicates non-UTF-8 source)
            if "�" in content:
                self.logger.warning(
                    f"[FileAgent] Non-UTF-8 bytes replaced with � while "
                    f"reading {filepath}"
                )
            return {"content": content}

        elif operation == "write":
            content = task.parameters.get("content", "")
            self._track_file_created(filepath)
            with open(filepath, "w", encoding="utf-8", errors="replace") as f:
                f.write(content)
            return {"message": "File written", "path": filepath}

        elif operation == "list":
            return {"files": os.listdir(filepath)}

        else:
            raise ValueError(f"Unknown operation: {operation}")
