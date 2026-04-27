from agents.base_agent import BaseAgent
from models import MemoryItem, Task


class MemoryAgent(BaseAgent):
    def __init__(self, memory_manager=None):
        super().__init__(
            name="MemoryAgent",
            agent_type="memory",
            description="Handles memory operations",
        )
        self.memory_manager = memory_manager

    def _do_work(self, task: Task):
        action = task.parameters.get("action")

        if action == "store":
            content = task.parameters.get("content")

            memory = MemoryItem.create_task_memory(
                content=content, task_id=task.task_id
            )

            if self.memory_manager:
                self.memory_manager.save(memory)

            return {"stored": content}

        elif action == "retrieve":
            query = task.parameters.get("query")

            if self.memory_manager:
                results = self.memory_manager.search(query)
                return {"results": results}

            return {"results": []}

        else:
            raise ValueError(f"Unknown action: {action}")
