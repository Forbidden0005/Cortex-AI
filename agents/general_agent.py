"""
General Agent - Handles open-ended, conversational, and unclassified requests.

Acts as the catch-all agent for requests that don't match any specialist
category (file, coding, web, etc.). Uses the LLM directly to generate a
response, making Cortex useful for plain conversation and Q&A as well as
structured tasks.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from core.llm_interface import LLMInterface
from models import Task


class GeneralAgent(BaseAgent):
    """
    Catch-all agent for conversational and unclassified requests.

    Routes to the LLM directly and returns the response as plain text.
    No external tools or file access required.
    """

    def __init__(self, llm: LLMInterface):
        super().__init__(
            name="GeneralAgent",
            agent_type="general",
            description="Handles open-ended and conversational requests via LLM",
        )
        self.llm = llm

    def _do_work(self, task: Task) -> dict:
        """
        Ask the LLM to respond to the user's request directly.

        Args:
            task: Task whose description is the user's message.

        Returns:
            Dict with "response" key containing the LLM's reply.
        """
        prompt = (
            "You are Cortex, a helpful AI assistant. "
            "Respond to the following request clearly and concisely.\n\n"
            f"Request: {task.description}"
        )

        response = self.llm.ask(prompt)
        return {"response": response}
