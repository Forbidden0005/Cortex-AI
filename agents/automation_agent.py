import subprocess

from agents.base_agent import BaseAgent
from models import Task


class AutomationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="AutomationAgent",
            agent_type="automation",
            description="Controls system automation",
        )

    def _do_work(self, task: Task):
        command = task.parameters.get("command")

        if not command:
            raise ValueError("No command provided")

        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        return {"stdout": result.stdout, "stderr": result.stderr}
