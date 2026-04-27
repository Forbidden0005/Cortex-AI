"""
Audio Agent - Processes audio files (transcription, analysis).

Currently a stub implementation. Intended to integrate with a
speech-to-text backend (e.g. Whisper) once the audio pipeline is wired up.

Expects task.parameters["audio_path"] to point to the audio file.
"""

from agents.base_agent import BaseAgent
from models import Task


class AudioAgent(BaseAgent):
    """
    Handles audio processing tasks such as transcription and analysis.

    Stub: returns a placeholder transcription. Replace _do_work with
    a real speech-to-text call when the audio backend is available.
    """

    def __init__(self):
        super().__init__(
            name="AudioAgent",
            agent_type="audio",
            description="Processes audio files (transcription, analysis)",
        )

    def _do_work(self, task: Task) -> dict:
        audio_path = task.parameters.get("audio_path")
        # TODO: integrate real speech-to-text backend (e.g. Whisper)
        return {"transcription": f"Processed audio: {audio_path}"}
