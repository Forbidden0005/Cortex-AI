from agents.base_agent import BaseAgent
from models import Task


class QAAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="QAAgent", agent_type="qa", description="Validates outputs"
        )

    def _do_work(self, task: Task):
        result = task.parameters.get("result")

        if result is None:
            raise ValueError("No result provided")

        return {"valid": True}
