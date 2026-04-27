from agents.base_agent import BaseAgent
from models import Task


class VisionAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="VisionAgent", agent_type="vision", description="Processes images"
        )

    def _do_work(self, task: Task):
        image_path = task.parameters.get("image_path")
        return {"analysis": f"Processed image: {image_path}"}
