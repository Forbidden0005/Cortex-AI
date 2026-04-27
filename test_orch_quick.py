"""Quick test of orchestrator initialization"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from agents.file_agent import FileAgent
from core.orchestrator import Orchestrator

print("Testing Orchestrator initialization...")

# Create orchestrator
orch = Orchestrator()
print("[OK] Orchestrator created")

# Register an agent
orch.agent_manager.register_agent(FileAgent())
print("[OK] Agent registered")

# Get stats
stats = orch.get_stats()
print(f"[OK] Stats retrieved: {stats['total_workflows']} workflows")

print("\n[PASS] Orchestrator working!")
