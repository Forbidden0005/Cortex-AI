"""
Terminal Tools - Terminal/shell operations for agents
"""

import sys
from pathlib import Path as PathLib

sys.path.append(str(PathLib(__file__).parent.parent))
from typing import Any, Dict, Optional

from core.logger import get_logger


class TerminalTools:
    """Terminal operation tools - wraps code_executor for consistency"""

    def __init__(self):
        self.logger = get_logger()
        # Import code executor
        from tools.code_executor import CodeExecutor

        self.executor = CodeExecutor()

    def run_command(
        self, command: str, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run shell command"""
        return self.executor.execute_shell(command, timeout)


if __name__ == "__main__":
    print("✅ Terminal Tools ready (uses CodeExecutor)")
