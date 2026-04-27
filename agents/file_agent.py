"""
File Agent - Handles filesystem read, write, and list operations.

Expects task.parameters to contain:
- "operation": one of "read", "write", "list"
- "filepath":  path to the target file or directory
- "content":   (write only) string content to write

When no filepath is provided, falls back to an LLM response if one
is available so freeform file-related questions are answered naturally.
"""

import os
from typing import Optional

from agents.base_agent import BaseAgent
from models import Task


class FileAgent(BaseAgent):
    """
    Performs basic filesystem operations.

    Supported operations:
    - read:  Read and return the contents of a file.
    - write: Write a string to a file (creates or overwrites).
    - list:  List the entries of a directory.

    If no filepath is present and an LLM was supplied, the agent
    answers conversationally rather than raising an error.
    """

    def __init__(self, llm=None):
        super().__init__(
            name="FileAgent",
            agent_type="file",
            description="Handles file read, write, and directory list operations",
        )
        self.llm = llm

    def _do_work(self, task: Task) -> dict:
        params = task.parameters or {}
        operation = params.get("operation")
        filepath = params.get("filepath")

        # No filepath — fall back to LLM for freeform file-related questions.
        if not filepath:
            if self.llm is not None:
                system_prompt = (
                    "You are Cortex, a helpful AI assistant. "
                    "Answer the user's file or folder question naturally. "
                    "If you need a specific path to act on, ask for it."
                )
                try:
                    response = self.llm.ask(
                        task.description,
                        system_prompt=system_prompt,
                        max_tokens=300,
                    ).strip()
                    return {"response": response}
                except Exception:
                    pass
            return {
                "response": (
                    "I can read, write, and list files. "
                    "Please give me a file or folder path to work with."
                )
            }

        # Explicit operation — default to "read" if not specified.
        op = operation or "read"

        if op == "read":
            self._track_file_read(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                return {"content": f.read()}

        elif op == "write":
            content = params.get("content", "")
            self._track_file_created(filepath)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return {"message": "File written", "path": filepath}

        elif op == "list":
            return {"files": os.listdir(filepath)}

        else:
            raise ValueError(f"Unknown operation: {op}")
