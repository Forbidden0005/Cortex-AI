from agents.base_agent import BaseAgent
from models import Task


class AudioAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="AudioAgent", agent_type="audio", description="Processes audio"
        )

    def _do_work(self, task: Task):
        audio_path = task.parameters.get("audio_path")
        return {"transcription": f"Processed audio: {audio_path}"}
