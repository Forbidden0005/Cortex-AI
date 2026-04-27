"""
Agent Manager - Manages all agents and routes tasks to the right one

Responsibilities:
- Register agents
- Route tasks to the correct agent based on task_type
- Track performance stats
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))
from core.logger import get_logger
from models import AgentResult, Task


class AgentManager:
    """
    Manages registered agents and routes tasks to the appropriate one.
    Falls back to 'file' agent if no exact match found.
    """

    def __init__(self):
        self.agents: Dict[str, Any] = {}  # agent_type -> BaseAgent
        self.logger = get_logger()

    def register_agent(self, agent) -> None:
        """Register an agent with the manager."""
        self.agents[agent.agent_type] = agent
        self.logger.info(
            f"[AgentManager] Registered: {agent.name} (type={agent.agent_type})"
        )

    def get_agent(self, agent_type: str) -> Optional[Any]:
        """Get an agent by type."""
        return self.agents.get(agent_type)

    def list_agents(self) -> List[str]:
        """Return list of registered agent names."""
        return [a.name for a in self.agents.values()]

    def execute_task(self, task: Task) -> AgentResult:
        """Route a task to the appropriate agent and execute it."""
        agent = self.agents.get(task.task_type)

        if not agent:
            self.logger.warning(
                f"[AgentManager] No agent for type '{task.task_type}', "
                f"falling back to 'file'"
            )
            agent = self.agents.get("file")
            if agent:
                # Remap task type so base_agent validation passes
                task.task_type = agent.agent_type

        if not agent:
            self.logger.error(
                f"[AgentManager] No agent available for task: {task.task_id}"
            )
            return AgentResult.create_error(
                error=f"No agent available for task type: {task.task_type}",
                agent_name="AgentManager",
                agent_type=task.task_type,
                task_id=task.task_id,
                task_description=task.description,
            )

        return agent.execute(task)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance stats for all registered agents."""
        return {
            agent_type: agent.get_stats() for agent_type, agent in self.agents.items()
        }
