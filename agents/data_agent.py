from agents.base_agent import BaseAgent
from models import Task


class DataAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DataAgent",
            agent_type="data",
            description="Processes and analyzes data",
        )

    def _do_work(self, task: Task):
        data = task.parameters.get("data", [])
        return {"count": len(data), "unique": len(set(data))}
