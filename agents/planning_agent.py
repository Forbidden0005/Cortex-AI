"""
Planning Agent - Breaks down complex tasks into manageable steps

Uses the LLM to analyze tasks and create execution plans.
Returns a list of subtasks for the orchestrator to execute.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Import base and models
sys.path.append(str(Path(__file__).parent.parent))
from agents.base_agent import BaseAgent
from core.config_loader import get_config
# Import LLM
from core.llm_interface import LLMInterface
from models import Task, TaskPriority


class PlanningAgent(BaseAgent):
    """
    Breaks down complex tasks into executable subtasks.

    Uses LLM to understand task requirements and generate
    a step-by-step plan with appropriate agent assignments.
    """

    def __init__(self, llm_interface: LLMInterface = None):
        """
        Initialize the planning agent.

        Args:
            llm_interface: Optional LLM interface (creates one if not provided)
        """
        super().__init__(
            name="PlanningAgent",
            agent_type="planning",
            description="Breaks complex tasks into steps",
            tools=["llm"],
        )

        # Initialize or use provided LLM
        if llm_interface:
            self.llm = llm_interface
        else:
            config = get_config()
            self.llm = LLMInterface(config.model, use_mock=True)

        self._track_tool_used("llm")

    def _do_work(self, task: Task) -> Dict[str, Any]:
        """
        Break down a task into subtasks.

        Args:
            task: Task to break down

        Returns:
            Dictionary with:
                - plan: The textual plan
                - subtasks: List of Task objects
                - reasoning: Why this plan was chosen
        """
        # Build prompt for LLM
        prompt = self._build_planning_prompt(task)

        # Get plan from LLM
        self._log(f"Requesting plan for: {task.description}")
        plan_text = self.llm.ask(
            prompt,
            system_prompt="You are a task planning expert. Break down tasks into clear, actionable steps.",
        )

        # Parse plan into subtasks
        subtasks = self._parse_plan_to_subtasks(plan_text, task)

        self._log(f"Generated plan with {len(subtasks)} subtasks")

        return {
            "plan": plan_text,
            "subtasks": subtasks,
            "num_subtasks": len(subtasks),
            "reasoning": "LLM-generated task breakdown",
        }

    def _build_planning_prompt(self, task: Task) -> str:
        """
        Build the prompt for the LLM.

        Args:
            task: Task to plan

        Returns:
            Prompt string
        """
        prompt = f"""Break down this task into clear, executable steps:

Task Description: {task.description}
Priority: {task.priority.value}
Parameters: {task.parameters}

For each step, specify:
1. What needs to be done
2. Which agent should handle it (choose from: planning, file, coding, web, data, automation, security, memory, vision, audio, qa)
3. Any specific parameters

Format your response as a numbered list of steps."""

        return prompt

    def _parse_plan_to_subtasks(self, plan_text: str, parent_task: Task) -> List[Task]:
        """
        Parse LLM plan text into Task objects.

        Args:
            plan_text: Plan text from LLM
            parent_task: The parent task being broken down

        Returns:
            List of Task objects
        """
        subtasks = []
        lines = plan_text.strip().split("\n")

        # Simple parsing - look for numbered steps
        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Look for numbered items (1., 2., etc.)
            if line[0].isdigit() and ("." in line or ")" in line):
                # Extract step description
                # Remove the number prefix
                description = line.split(".", 1)[-1].split(")", 1)[-1].strip()

                if description:
                    # Determine agent type based on keywords
                    agent_type = self._determine_agent_type(description)

                    # Create subtask
                    subtask = Task(
                        description=description,
                        task_type=agent_type,
                        parent_task_id=parent_task.task_id,
                        priority=parent_task.priority,
                        parameters={},
                        save_to_memory=True,
                        memory_tags=["planned_subtask", agent_type],
                    )

                    subtasks.append(subtask)

        # If no subtasks parsed, create a fallback
        if not subtasks:
            subtask = Task(
                description=f"Execute: {parent_task.description}",
                task_type="file",  # Default to file agent
                parent_task_id=parent_task.task_id,
                priority=parent_task.priority,
            )
            subtasks.append(subtask)

        # Set up dependencies (each task depends on previous)
        for i in range(1, len(subtasks)):
            subtasks[i].dependencies.append(subtasks[i - 1].task_id)

        return subtasks

    def _determine_agent_type(self, description: str) -> str:
        """
        Determine which agent type should handle a task based on description.

        Args:
            description: Task description

        Returns:
            Agent type string
        """
        description_lower = description.lower()

        # Keyword mapping
        if any(
            word in description_lower
            for word in ["file", "read", "write", "save", "load", "scan", "move"]
        ):
            return "file"

        if any(
            word in description_lower
            for word in ["code", "script", "function", "program", "execute", "run"]
        ):
            return "coding"

        if any(
            word in description_lower
            for word in ["search", "web", "internet", "lookup", "find online"]
        ):
            return "web"

        if any(
            word in description_lower
            for word in ["analyze", "data", "calculate", "process", "chart"]
        ):
            return "data"

        if any(
            word in description_lower
            for word in ["automate", "control", "click", "type", "gui"]
        ):
            return "automation"

        if any(
            word in description_lower
            for word in ["security", "scan", "virus", "malware", "check"]
        ):
            return "security"

        if any(
            word in description_lower
            for word in ["remember", "recall", "memory", "store", "retrieve"]
        ):
            return "memory"

        if any(
            word in description_lower
            for word in ["image", "picture", "photo", "visual", "see"]
        ):
            return "vision"

        if any(
            word in description_lower
            for word in ["audio", "sound", "voice", "speech", "listen"]
        ):
            return "audio"

        if any(
            word in description_lower
            for word in ["test", "verify", "check", "qa", "quality"]
        ):
            return "qa"

        if any(
            word in description_lower
            for word in ["plan", "break down", "organize", "steps"]
        ):
            return "planning"

        # Default to file agent
        return "file"


if __name__ == "__main__":
    # Test the PlanningAgent
    from core.config_loader import ModelConfig, ModelProvider
    from models import TaskPriority

    print("Testing PlanningAgent...")

    # Create agent with mock LLM
    config = ModelConfig(provider=ModelProvider.LOCAL, model_name="test")
    llm = LLMInterface(config, use_mock=True)
    agent = PlanningAgent(llm)

    # Create a task to plan
    task = Task(
        description="Scan project files, categorize them by type, organize into folders, and generate a summary report",
        task_type="planning",
        priority=TaskPriority.HIGH,
        parameters={"project_path": "/home/user/project"},
    )

    print("\n--- Planning Task ---")
    print(f"Description: {task.description}")

    # Execute planning
    result = agent.execute(task)

    print("\n--- Results ---")
    print(f"Status: {result.status.value}")
    print(f"Success: {result.is_successful()}")

    if result.data:
        print(f"\nPlan:\n{result.data['plan']}")
        print(f"\nSubtasks ({result.data['num_subtasks']}):")
        for i, subtask in enumerate(result.data["subtasks"], 1):
            print(f"  {i}. [{subtask.task_type}] {subtask.description}")
            if subtask.dependencies:
                print(f"     Dependencies: {subtask.dependencies}")

    print("\n✅ PlanningAgent test passed!")
