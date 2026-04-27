"""
Vision Agent - Processes images (object detection, OCR, captioning).

Currently a stub. Intended to integrate with a vision model
(e.g. LLaVA, CLIP, or a hosted vision API).

Expects task.parameters["image_path"] to point to the image file.
"""

from agents.base_agent import BaseAgent
from models import Task


class VisionAgent(BaseAgent):
    """
    Analyzes images and returns structured descriptions.

    Stub: returns a placeholder analysis. Replace _do_work with a
    real vision model call when the vision backend is available.
    """

    def __init__(self):
        super().__init__(
            name="VisionAgent",
            agent_type="vision",
            description="Processes images (object detection, OCR, captioning)",
        )

    def _do_work(self, task: Task) -> dict:
        image_path = task.parameters.get("image_path")
        # TODO: integrate real vision model (e.g. LLaVA, CLIP)
        return {"analysis": f"Processed image: {image_path}"}
