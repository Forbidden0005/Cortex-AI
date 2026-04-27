import subprocess
import tempfile

from agents.base_agent import BaseAgent
from models import Task


class CodingAgent(BaseAgent):
    def __init__(self, llm_interface=None):
        super().__init__(
            name="CodingAgent",
            agent_type="coding",
            description="Generates and executes code",
        )
        self.llm = llm_interface

    def _do_work(self, task: Task):
        action = task.parameters.get("action")

        if action == "generate":
            prompt = task.description
            code = self.llm.ask(prompt) if self.llm else "# No LLM connected"
            return {"code": code}

        elif action == "execute":
            code = task.parameters.get("code")

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                path = f.name

            self._track_file_created(path)

            result = subprocess.run(["python", path], capture_output=True, text=True)

            return {"stdout": result.stdout, "stderr": result.stderr}

        else:
            raise ValueError(f"Unknown action: {action}")
