"""
AI System - Main Entry Point

Initializes all components and provides interface for users.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from agents.audio_agent import AudioAgent
from agents.automation_agent import AutomationAgent
from agents.coding_agent import CodingAgent
from agents.data_agent import DataAgent
from agents.file_agent import FileAgent
from agents.memory_agent import MemoryAgent
# Import all agents
from agents.planning_agent import PlanningAgent
from agents.qa_agent import QAAgent
from agents.security_agent import SecurityAgent
from agents.vision_agent import VisionAgent
from agents.web_agent import WebAgent
from core.logger import get_logger
from core.orchestrator import Orchestrator
from core.reflection import ReflectionSystem


class AISystem:
    """
    Main AI System class.

    Initializes all components and provides a clean interface.
    """

    def __init__(self):
        """Initialize the AI system"""
        print("=" * 60)
        print("AI SYSTEM INITIALIZATION")
        print("=" * 60)

        self.logger = get_logger()
        self.logger.info("Starting AI System initialization...")

        # Initialize orchestrator (includes LLM, memory, etc.)
        print("\n[1/3] Initializing Orchestrator...")
        self.orchestrator = Orchestrator()
        print("✓ Orchestrator ready")

        # Initialize reflection system
        print("\n[2/3] Initializing Reflection System...")
        self.reflection = ReflectionSystem(self.orchestrator.memory)
        print("✓ Reflection system ready")

        # Register all agents
        print("\n[3/3] Registering Agents...")
        self._register_agents()
        print(
            f"✓ Registered {len(self.orchestrator.agent_manager.list_agents())} agents"
        )

        print("\n" + "=" * 60)
        print("✅ AI SYSTEM READY")
        print("=" * 60)

        self.logger.log_event(
            "system_initialized", "AI System fully initialized and ready"
        )

    def _register_agents(self):
        """Register all agents with the orchestrator"""
        agents = [
            PlanningAgent(self.orchestrator.llm),
            FileAgent(),
            CodingAgent(self.orchestrator.llm),
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
            self.orchestrator.agent_manager.register_agent(agent)
            print(f"  ✓ {agent.name}")

    def process(self, user_request: str, context: dict = None) -> dict:
        """
        Process a user request.

        Args:
            user_request: Natural language request
            context: Optional context dictionary

        Returns:
            Dictionary with results
        """
        self.logger.info(f"Processing request: {user_request}")

        # Process through orchestrator
        result = self.orchestrator.process_request(user_request, context)

        # If successful, reflect on execution
        if result["success"]:
            # Get the tasks that were executed
            workflow = self.orchestrator.get_workflow_status(result["workflow_id"])
            if workflow and "results" in workflow:
                for agent_result in workflow["results"]:
                    self.reflection.reflect_on_task(
                        task_description=agent_result.task_description,
                        success=agent_result.is_successful(),
                        notes=agent_result.error or "",
                    )

        return result

    def get_stats(self) -> dict:
        """Get system statistics"""
        orch_stats = self.orchestrator.get_stats()
        reflection_insights = self.reflection.get_insights()

        return {
            "orchestrator": orch_stats,
            "reflection": reflection_insights,
            "suggestions": self.reflection.suggest_improvements(),
        }

    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)
        print("Type your requests below. Type 'quit' or 'exit' to stop.")
        print("Type 'stats' to see system statistics.")
        print("=" * 60 + "\n")

        while True:
            try:
                # Get user input
                user_input = input("\n🤖 You: ").strip()

                if not user_input:
                    continue

                # Check for commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == "stats":
                    stats = self.get_stats()
                    print("\n📊 System Statistics:")
                    print(
                        f"   Workflows: {stats['orchestrator']['completed_workflows']} completed"
                    )
                    print(
                        f"   Tasks: {stats['orchestrator']['queue_stats']['completed']} completed"
                    )
                    print(
                        f"   Success Rate: {stats['orchestrator']['queue_stats']['success_rate']:.1%}"
                    )
                    print("\n💡 Suggestions:")
                    for suggestion in stats["suggestions"]:
                        print(f"   - {suggestion}")
                    continue

                # Process request
                print("\n⚙️  Processing...")
                result = self.process(user_input)

                if result["success"]:
                    print("\n✅ Success!")
                    if isinstance(result["result"], dict):
                        if "summary" in result["result"]:
                            print(f"   {result['result']['summary']}")
                        else:
                            print(f"   Result: {str(result['result'])[:200]}")
                    else:
                        print(f"   Result: {str(result['result'])[:200]}")
                else:
                    print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    """Main entry point"""
    # Initialize system
    system = AISystem()

    # Check if running interactively
    if len(sys.argv) > 1:
        # Process command line request
        request = " ".join(sys.argv[1:])
        print(f"\nProcessing: {request}\n")
        result = system.process(request)

        if result["success"]:
            print(f"\n✅ Success: {result.get('message', 'Done')}")
            print(f"Result: {result['result']}")
        else:
            print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")
    else:
        # Run interactive mode
        system.interactive_mode()


if __name__ == "__main__":
    main()
