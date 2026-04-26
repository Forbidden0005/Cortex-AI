"""
Agent System Integration Test

Tests all agents with the AgentManager and Task Queue
"""

import io
import sys
from pathlib import Path

# Ensure stdout can handle Unicode on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.append(str(Path(__file__).parent))

from agents.audio_agent import AudioAgent
from agents.automation_agent import AutomationAgent
from agents.coding_agent import CodingAgent
from agents.data_agent import DataAgent
# Import all agents
from agents.file_agent import FileAgent
from agents.memory_agent import MemoryAgent
from agents.planning_agent import PlanningAgent
from agents.qa_agent import QAAgent
from agents.security_agent import SecurityAgent
from agents.vision_agent import VisionAgent
from agents.web_agent import WebAgent
from core.agent_manager import AgentManager
from core.llm_interface import LLMInterface, ModelConfig, ModelProvider
from core.task_queue import TaskQueue
from models import Task, TaskPriority


def test_all_agents():
    """Test all agents are properly configured"""
    print("=" * 60)
    print("AGENT SYSTEM INTEGRATION TEST")
    print("=" * 60)

    # Create LLM interface (mock for testing)
    print("\n[1/4] Creating LLM Interface...")
    config = ModelConfig(provider=ModelProvider.LOCAL, model_name="test")
    llm = LLMInterface(config, use_mock=True)
    print("✓ LLM Interface created")

    # Create agent manager
    print("\n[2/4] Creating Agent Manager...")
    manager = AgentManager()
    print("✓ Agent Manager created")

    # Register all agents
    print("\n[3/4] Registering Agents...")
    agents = [
        FileAgent(),
        CodingAgent(llm),
        PlanningAgent(llm),
        DataAgent(),
        WebAgent(),
        AutomationAgent(),
        SecurityAgent(),
        MemoryAgent(),
        VisionAgent(),
        AudioAgent(),
        QAAgent(),
    ]

    for agent in agents:
        manager.register_agent(agent)
        print(f"  ✓ {agent.name} ({agent.agent_type})")

    print(f"\n✓ Registered {len(agents)} agents")

    # Test each agent with a simple task
    print("\n[4/4] Testing Each Agent...")

    test_tasks = [
        Task(
            description="Test file operations",
            task_type="file",
            parameters={"operation": "list", "filepath": "."},
        ),
        Task(
            description="Generate Python code",
            task_type="coding",
            parameters={"action": "generate"},
        ),
        Task(description="Create a plan", task_type="planning"),
        Task(
            description="Analyze data",
            task_type="data",
            parameters={"action": "analyze", "data": [1, 2, 3]},
        ),
        Task(
            description="Search the web",
            task_type="web",
            parameters={"action": "search", "query": "test"},
        ),
        Task(
            description="Automate task",
            task_type="automation",
            parameters={"action": "click", "target": "button"},
        ),
        Task(
            description="Security scan",
            task_type="security",
            parameters={"action": "scan", "target": "file.txt"},
        ),
        Task(
            description="Store memory",
            task_type="memory",
            parameters={"action": "store", "content": "test"},
        ),
        Task(
            description="Process image",
            task_type="vision",
            parameters={"action": "analyze", "image": "test.jpg"},
        ),
        Task(
            description="Process audio",
            task_type="audio",
            parameters={"action": "transcribe", "audio": "test.mp3"},
        ),
        Task(
            description="Quality check",
            task_type="qa",
            parameters={"action": "test", "target": "code"},
        ),
    ]

    results = []
    for task in test_tasks:
        result = manager.execute_task(task)
        status = "✓" if result.is_successful() else "✗"
        results.append(result.is_successful())
        print(f"  {status} {result.agent_name}: {result.status.value}")
        if not result.is_successful():
            print(f"      Error: {result.error}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"Total Agents: {len(agents)}")
    print(f"Tests Run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    # Performance stats
    print("\n" + "=" * 60)
    print("AGENT PERFORMANCE")
    print("=" * 60)

    stats = manager.get_performance_stats()
    for agent_name, metrics in sorted(stats.items()):
        print(f"\n{agent_name}:")
        print(f"  Tasks: {metrics.get('total_tasks', 0)}")
        print(f"  Success Rate: {metrics.get('success_rate', 0):.1%}")
        avg = metrics.get('avg_time', metrics.get('average_time', 0))
        print(f"  Avg Time: {avg:.4f}s")

    # Final result
    print("\n" + "=" * 60)
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"⚠️  {failed} TESTS FAILED")
    print("=" * 60)

    return failed == 0


def test_task_queue_integration():
    """Test Task Queue with Agent Manager"""
    print("\n\n" + "=" * 60)
    print("TASK QUEUE INTEGRATION TEST")
    print("=" * 60)

    # Create components
    print("\nInitializing...")
    manager = AgentManager()
    queue = TaskQueue()

    # Register a few agents
    manager.register_agent(FileAgent())
    manager.register_agent(PlanningAgent())

    print("✓ Components initialized")

    # Create tasks with dependencies
    print("\nCreating dependent tasks...")
    task1 = Task(
        description="First task",
        task_type="file",
        priority=TaskPriority.HIGH,
        parameters={"operation": "list", "filepath": "."},
    )

    task2 = Task(
        description="Second task (depends on first)",
        task_type="planning",
        priority=TaskPriority.MEDIUM,
        dependencies=[task1.task_id],
    )

    # Add to queue
    queue.add_task(task1)
    queue.add_task(task2)

    print("✓ Added 2 tasks to queue")

    # Process queue
    print("\nProcessing queue...")
    processed = 0

    while not queue.is_empty():
        task = queue.get_next_task()
        if not task:
            break

        print(f"  Executing: {task.description}")
        result = manager.execute_task(task)

        if result.is_successful():
            queue.mark_completed(task, result.data)
            print(f"    ✓ Completed in {result.execution_time:.4f}s")
        else:
            queue.mark_failed(task, result.error)
            print(f"    ✗ Failed: {result.error}")

        processed += 1

    # Summary
    stats = queue.get_stats()
    print(f"\n✓ Processed {processed} tasks")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")

    print("\n" + "=" * 60)
    print("✅ TASK QUEUE INTEGRATION TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests
    success = test_all_agents()
    test_task_queue_integration()

    if success:
        print("\n\n🎉 ALL SYSTEMS OPERATIONAL! 🎉")
    else:
        print("\n\n⚠️  Some tests failed - check logs above")
