"""
Reflection System - Analyses past executions and generates insights

Stores reflections in memory and surfaces patterns to help the
system (and user) understand what's working and what isn't.
"""

import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent))
from core.logger import get_logger
from models import MemoryImportance, MemoryItem, MemoryType


class ReflectionSystem:
    """
    Analyses completed workflows and generates human-readable insights
    and improvement suggestions based on patterns in memory.
    """

    def __init__(self, memory):
        """
        Args:
            memory: MemoryManager instance (injected from Orchestrator)
        """
        self.memory = memory
        self.logger = get_logger()

    def reflect_on_task(
        self, task_description: str, success: bool, notes: str = ""
    ) -> None:
        """Persist a reflection about a single completed task."""
        status = "Success" if success else "Failure"
        content = f"Task: {task_description} | Status: {status}" + (
            f" | Notes: {notes}" if notes else ""
        )
        item = MemoryItem(
            content=content,
            memory_type=MemoryType.REFLECTION,
            importance=MemoryImportance.MEDIUM,
            tags=["reflection", "task", status.lower()],
            source="reflection_system",
        )
        self.memory.save(item)
        self.logger.info(f"[Reflection] Stored reflection: {status}")

    def get_insights(self) -> List[str]:
        """Return a list of insight strings based on stored reflections."""
        reflections = self.memory.get_by_type(MemoryType.REFLECTION)

        if not reflections:
            return [
                "No reflections recorded yet — run some tasks to generate insights."
            ]

        total = len(reflections)
        successes = sum(1 for r in reflections if "Status: Success" in r.content)
        failures = total - successes
        rate = successes / total if total > 0 else 0.0

        insights = [
            f"Total tasks reflected on: {total}",
            f"Successes: {successes}  |  Failures: {failures}",
            f"Overall success rate: {rate:.1%}",
        ]

        if failures > 0:
            insights.append(f"{failures} task(s) failed — review logs for patterns.")

        return insights

    def suggest_improvements(self) -> List[str]:
        """Generate actionable improvement suggestions."""
        reflections = self.memory.get_by_type(MemoryType.REFLECTION)
        suggestions = []

        if not reflections:
            suggestions.append(
                "Run more tasks to enable data-driven improvement suggestions."
            )
            return suggestions

        failures = [r for r in reflections if "Status: Failure" in r.content]
        if failures:
            suggestions.append(
                f"Review {len(failures)} failed task(s) to identify common error patterns."
            )

        if len(reflections) > 50:
            suggestions.append(
                "Memory is getting large — consider running cleanup_old_memories()."
            )

        if not suggestions:
            suggestions.append(
                "System is performing well. Keep monitoring for patterns."
            )

        return suggestions
