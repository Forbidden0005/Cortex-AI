"""
Automation Agent - Executes shell commands and system automation tasks.

Accepts a pre-tokenized command list via task.parameters["command"] to
avoid shell injection. Passing a plain string is also supported but
will be split on whitespace — use a list for commands with arguments
containing spaces.
"""

import shlex
import subprocess
from typing import List, Union

from agents.base_agent import BaseAgent
from models import Task


class AutomationAgent(BaseAgent):
    """
    Runs system commands in a subprocess.

    Expects task.parameters["command"] to be either:
    - A list of strings: ["ls", "-la", "/tmp"]  (preferred, injection-safe)
    - A plain string: "ls -la /tmp"  (split on whitespace automatically)

    Returns stdout and stderr from the process.
    """

    def __init__(self):
        super().__init__(
            name="AutomationAgent",
            agent_type="automation",
            description="Executes shell commands safely without shell=True",
        )

    def _do_work(self, task: Task) -> dict:
        command: Union[str, List[str], None] = task.parameters.get("command")

        if not command:
            raise ValueError("No command provided")

        # Normalise to a list so we never need shell=True
        if isinstance(command, str):
            args = shlex.split(command)
        else:
            args = list(command)

        self._track_tool_used("subprocess")
        result = subprocess.run(args, shell=False, capture_output=True, text=True)

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
