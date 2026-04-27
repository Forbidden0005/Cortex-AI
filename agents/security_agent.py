"""
Security Agent - Performs security checks on files and paths.

Currently checks for file existence. Intended to grow into a full
security analysis agent (permissions, virus scan, secrets detection).

Expects task.parameters["filepath"] to be the path to inspect.
When no filepath is provided, falls back to an LLM response if one
is available.
"""

import os
from typing import Optional

from agents.base_agent import BaseAgent
from models import Task


class SecurityAgent(BaseAgent):
    """
    Runs security checks on files and system paths.

    Stub: verifies that the target path exists. Extend _do_work to add
    deeper checks such as permission auditing, secrets scanning, or
    antivirus integration.

    If no filepath is present and an LLM was supplied, the agent
    answers conversationally rather than raising an error.
    """

    def __init__(self, llm=None):
        super().__init__(
            name="SecurityAgent",
            agent_type="security",
            description="Performs security checks on files and paths",
        )
        self.llm = llm

    def _do_work(self, task: Task) -> dict:
        filepath = task.parameters.get("filepath")

        if not filepath:
            if self.llm is not None:
                system_prompt = (
                    "You are Cortex, a helpful AI assistant with security expertise. "
                    "Answer the user's security question naturally and concisely. "
                    "If you can't perform the action directly, explain what you can do "
                    "and ask for any details you need (e.g. a specific path to inspect)."
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
                    "I can run security checks on files and directories. "
                    "Please tell me which path you'd like me to inspect."
                )
            }

        # TODO: add deeper checks (permissions, secrets, antivirus)
        exists = os.path.exists(filepath)
        return {
            "path": filepath,
            "exists": exists,
            "response": (
                f"'{filepath}' exists on disk."
                if exists
                else f"'{filepath}' was not found."
            ),
        }
