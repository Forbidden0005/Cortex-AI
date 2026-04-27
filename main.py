"""
Cortex - Main Entry Point

Initializes all components and provides interface for users.
"""

import itertools
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from agents.audio_agent import AudioAgent
from agents.automation_agent import AutomationAgent
from agents.coding_agent import CodingAgent
from agents.data_agent import DataAgent
from agents.file_agent import FileAgent
from agents.general_agent import GeneralAgent
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


_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_TERMINAL_WIDTH = 78  # safe width for most terminals


class Spinner:
    """
    Lightweight terminal spinner for simple/fast responses.

    Usage::
        with Spinner("Thinking"):
            result = slow_call()
    """

    def __init__(self, label: str = "Thinking"):
        self.label = label
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        for frame in itertools.cycle(_SPINNER_FRAMES):
            if self._stop_event.is_set():
                break
            sys.stdout.write(f"\r{frame}  {self.label}...")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r" + " " * (_TERMINAL_WIDTH) + "\r")
        sys.stdout.flush()

    def __enter__(self) -> "Spinner":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop_event.set()
        self._thread.join()


class LiveStatus:
    """
    Live status bar for complex multi-step tasks.

    Displays the current step, elapsed time, and a spinner — all on a
    single line that rewrites itself. Call update() from any thread to
    change the status message.

    Usage::
        with LiveStatus() as status:
            result = orchestrator.process_request(
                ..., on_status=status.update
            )
    """

    def __init__(self):
        self._message = "Starting..."
        self._lock = threading.Lock()
        self._start = time.time()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def update(self, message: str) -> None:
        """Thread-safe status update."""
        with self._lock:
            self._message = message

    def _run(self) -> None:
        for frame in itertools.cycle(_SPINNER_FRAMES):
            if self._stop_event.is_set():
                break
            elapsed = time.time() - self._start
            with self._lock:
                msg = self._message
            # Truncate so the line never wraps
            status = f"{frame}  {msg}  [{elapsed:.1f}s]"
            if len(status) > _TERMINAL_WIDTH:
                status = status[: _TERMINAL_WIDTH - 3] + "..."
            sys.stdout.write(f"\r{status:<{_TERMINAL_WIDTH}}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * _TERMINAL_WIDTH + "\r")
        sys.stdout.flush()

    def __enter__(self) -> "LiveStatus":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop_event.set()
        self._thread.join()


class Cortex:
    """
    Cortex - Multi-agent AI orchestration system.

    Initializes all components and provides a clean interface.
    """

    def __init__(self):
        """Initialize Cortex"""
        print("=" * 60)
        print("CORTEX INITIALIZATION")
        print("=" * 60)

        self.logger = get_logger()
        self.logger.info("Starting Cortex initialization...")

        # Initialize orchestrator (includes LLM, memory, etc.)
        print("\n[1/3] Initializing Orchestrator...")
        self.orchestrator = Orchestrator()
        print("[OK] Orchestrator ready")

        # Initialize reflection system
        print("\n[2/3] Initializing Reflection System...")
        self.reflection = ReflectionSystem(self.orchestrator.memory)
        print("[OK] Reflection system ready")

        # Register all agents
        print("\n[3/3] Registering Agents...")
        self._register_agents()
        print(
            f"[OK] Registered {len(self.orchestrator.agent_manager.list_agents())} agents"
        )

        print("\n" + "=" * 60)
        print("CORTEX READY")
        print("=" * 60)

        self.logger.log_event(
            "system_initialized", "Cortex fully initialized and ready"
        )

    def _register_agents(self):
        """Register all agents with the orchestrator"""
        agents = [
            PlanningAgent(self.orchestrator.llm),
            FileAgent(self.orchestrator.llm),
            CodingAgent(self.orchestrator.llm),
            DataAgent(self.orchestrator.llm),
            WebAgent(self.orchestrator.llm),
            AutomationAgent(),
            SecurityAgent(self.orchestrator.llm),
            MemoryAgent(),
            VisionAgent(),
            AudioAgent(),
            QAAgent(),
            GeneralAgent(self.orchestrator.llm),
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

        # If successful, reflect on execution (skip fast-path direct responses)
        if result["success"] and result.get("workflow_id"):
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

                # Classify once; reuse the result so process_request doesn't
                # classify again (avoids a redundant LLM call for every message).
                task_type = self.orchestrator._classify_request(user_input)
                is_simple = task_type == "general"

                if is_simple:
                    with Spinner("Thinking"), self.logger.quiet_console():
                        result = self.orchestrator.process_request(
                            user_input, _task_type=task_type
                        )
                else:
                    with LiveStatus() as status, self.logger.quiet_console():
                        result = self.orchestrator.process_request(
                            user_input,
                            _task_type=task_type,
                            on_status=status.update,
                        )

                if result["success"]:
                    res = result.get("result") or {}
                    if isinstance(res, dict) and "response" in res:
                        # Conversational / fallback LLM reply
                        print(f"Cortex: {res['response']}")
                    elif isinstance(res, dict) and "summary" in res:
                        # Multi-step task summary
                        print(f"\n[OK] {res['summary']}")
                    elif isinstance(res, dict) and res:
                        # Some other structured result — show as compact key=value
                        parts = ", ".join(f"{k}={v}" for k, v in res.items())
                        print(f"\n[OK] {parts[:300]}")
                    else:
                        # Empty or unrecognised — don't print None
                        print("\n[OK] Done.")
                else:
                    print(f"\nCortex: Sorry, I ran into a problem — {result.get('error', 'unknown error')}.")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    """Main entry point"""
    # Initialize system
    system = Cortex()

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
