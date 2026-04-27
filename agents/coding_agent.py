"""
Coding Agent - Generates, reviews, and executes Python code.

Supports three explicit actions via task.parameters["action"]:
  generate  -- write code from a description
  execute   -- run a provided code snippet
  review    -- critique existing code

Falls back to a direct LLM response for any other request (e.g. a
conversational question about coding that slipped past the classifier).
"""

import subprocess
import tempfile

from agents.base_agent import BaseAgent
from models import Task


class CodingAgent(BaseAgent):
    """Generates, executes, and reviews Python code."""

    def __init__(self, llm_interface=None):
        super().__init__(
            name="CodingAgent",
            agent_type="coding",
            description="Generates and executes code",
        )
        self.llm = llm_interface

    def _do_work(self, task: Task) -> dict:
        action = task.parameters.get("action")

        if action == "generate":
            code = self.llm.ask(task.description) if self.llm else "# No LLM connected"
            return {"code": code}

        elif action == "execute":
            code = task.parameters.get("code", "")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                path = f.name
            self._track_file_created(path)
            result = subprocess.run(["python", path], capture_output=True, text=True)
            return {"stdout": result.stdout, "stderr": result.stderr}

        elif action == "review":
            code = task.parameters.get("code", task.description)
            prompt = f"Review this Python code and give concise feedback:\n\n{code}"
            feedback = self.llm.ask(prompt) if self.llm else "No LLM connected"
            return {"review": feedback}

        else:
            # Fallback: treat as a general coding question and answer via LLM.
            # This handles requests that the classifier routed here but that
            # don't require code generation or execution.
            if self.llm:
                system = (
                    "You are Cortex, an expert Python developer. "
                    "Answer the following coding question clearly and concisely."
                )
                response = self.llm.ask(task.description, system_prompt=system)
                return {"response": response}
            raise ValueError(f"Unknown action: {action}")
