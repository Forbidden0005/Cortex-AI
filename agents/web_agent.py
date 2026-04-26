from agents.base_agent import BaseAgent
from models import Task


class WebAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="WebAgent", agent_type="web", description="Handles web operations"
        )

    def _do_work(self, task: Task):
        query = task.parameters.get("query")
        return {"result": f"Search result placeholder for: {query}"}
