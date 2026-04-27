"""
Data Agent - Processes, analyzes, and summarizes structured data.

Currently provides basic count and uniqueness statistics. Intended to
grow into a full data analysis agent (pandas, charting, aggregation).

Expects task.parameters["data"] to be a list of values. When no data
is provided, falls back to an LLM response if one is available.
"""

from typing import Optional

from agents.base_agent import BaseAgent
from models import Task


class DataAgent(BaseAgent):
    """
    Analyzes structured data passed in task parameters.

    Returns basic summary statistics. Extend _do_work to support
    richer operations such as aggregation, filtering, and charting.

    If no data is present in task.parameters and an LLM was supplied,
    the agent answers conversationally rather than returning empty stats.
    """

    def __init__(self, llm=None):
        super().__init__(
            name="DataAgent",
            agent_type="data",
            description="Processes and analyzes structured data",
        )
        self.llm = llm  # Optional — used for fallback when no data provided

    def _do_work(self, task: Task) -> dict:
        data = task.parameters.get("data", [])

        # If no data was passed, fall back to an LLM response so we don't
        # return a useless {'count': 0, 'unique': 0} for conversational inputs.
        if not data:
            if self.llm is not None:
                system_prompt = (
                    "You are Cortex, a helpful AI assistant specializing in data analysis. "
                    "Answer the user's question or request naturally and concisely."
                )
                try:
                    response = self.llm.ask(
                        task.description,
                        system_prompt=system_prompt,
                        max_tokens=300,
                    ).strip()
                    return {"response": response}
                except Exception:
                    pass
            return {
                "response": (
                    "I can analyze data for you, but I need a dataset to work with. "
                    "Please provide data in your request."
                )
            }

        # TODO: add richer analytics (groupby, stats, chart generation)
        return {"count": len(data), "unique": len(set(data))}
