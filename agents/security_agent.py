import os

from agents.base_agent import BaseAgent
from models import Task


class SecurityAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SecurityAgent",
            agent_type="security",
            description="Performs security checks",
        )

    def _do_work(self, task: Task):
        filepath = task.parameters.get("filepath")

        if not filepath:
            raise ValueError("No filepath provided")

        return {"exists": os.path.exists(filepath)}
