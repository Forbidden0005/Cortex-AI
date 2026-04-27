"""
QA Agent - Validates and quality-checks the output of other agents.

Accepts a result object via task.parameters["result"] and applies
validation rules. Currently performs a basic non-null check; extend
with schema validation, assertion rules, or LLM-based review as needed.
"""

from agents.base_agent import BaseAgent
from models import Task


class QAAgent(BaseAgent):
    """
    Validates agent outputs against quality criteria.

    Raises ValueError if the result is missing. Extend _do_work to add
    richer validation such as schema checks, content rules, or scoring.
    """

    def __init__(self):
        super().__init__(
            name="QAAgent",
            agent_type="qa",
            description="Validates and quality-checks outputs from other agents",
        )

    def _do_work(self, task: Task) -> dict:
        result = task.parameters.get("result")

        if result is None:
            raise ValueError("No result provided")

        # TODO: add richer validation rules (schema, content checks, scoring)
        return {"valid": True}
