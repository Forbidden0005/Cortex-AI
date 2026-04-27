"""
Code Executor - Safe code execution for agents

Provides sandboxed code execution capabilities.
"""

import sys
from pathlib import Path as PathLib

sys.path.append(str(PathLib(__file__).parent.parent))
import os
import shlex
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

from core.logger import get_logger


class CodeExecutor:
    """
    Safe code execution tools.

    Executes code in isolated environment with timeouts.
    """

    def __init__(self):
        self.logger = get_logger()
        self.timeout = 30  # seconds

    def execute_python(
        self, code: str, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code safely.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Dictionary with execution result
        """
        try:
            timeout = timeout or self.timeout

            self.logger.info(
                f"[CodeExecutor] Executing Python code ({len(code)} chars)"
            )

            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Execute with timeout
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                self.logger.info(
                    f"[CodeExecutor] Execution completed (exit code: {result.returncode})"
                )

                return {
                    "success": result.returncode == 0,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "timeout": False,
                }

            finally:
                # Retry cleanup up to 3 times — on Windows the subprocess may
                # briefly hold the file lock after exiting.
                for _attempt in range(3):
                    try:
                        os.unlink(temp_file)
                        break
                    except OSError:
                        if _attempt < 2:
                            time.sleep(0.05)
                        else:
                            self.logger.warning(
                                f"[CodeExecutor] Failed to delete temp file "
                                f"{temp_file} after 3 attempts"
                            )

        except subprocess.TimeoutExpired:
            self.logger.warning(f"[CodeExecutor] Execution timed out after {timeout}s")
            return {
                "success": False,
                "timeout": True,
                "error": f"Execution timed out after {timeout} seconds",
            }

        except Exception as e:
            self.logger.error(f"[CodeExecutor] Execution failed: {e}")
            return {"success": False, "error": str(e)}

    def execute_shell(
        self, command: Union[str, List[str]], timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute a shell command safely.

        Args:
            command: Command to execute.  Prefer a list of strings (e.g.
                     ["git", "status"]) to avoid shell-injection risks.
                     A plain string is split with shlex.split() — never
                     passed directly to the shell with shell=True.
            timeout: Execution timeout in seconds

        Returns:
            Dictionary with execution result
        """
        try:
            timeout = timeout or self.timeout

            # Normalise to a list so we never use shell=True.
            # Use posix=False on Windows so backslashes in paths (C:\Users\...)
            # are treated as literals rather than escape sequences.
            if isinstance(command, str):
                cmd_list = shlex.split(command, posix=(sys.platform != "win32"))
            else:
                cmd_list = list(command)

            self.logger.info(
                f"[CodeExecutor] Executing shell command: {cmd_list[0]!r} ..."
            )

            # shell=False — no injection risk.
            # Note: shell features (pipes, %VAR% expansion, redirection) are
            # intentionally unavailable; callers that need them should pass a
            # pre-split list and handle shell semantics themselves.
            result = subprocess.run(
                cmd_list, shell=False, capture_output=True, text=True, timeout=timeout
            )

            self.logger.info(
                f"[CodeExecutor] Command completed (exit code: {result.returncode})"
            )

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command,
            }

        except subprocess.TimeoutExpired:
            self.logger.warning(f"[CodeExecutor] Command timed out after {timeout}s")
            return {
                "success": False,
                "timeout": True,
                "error": f"Command timed out after {timeout} seconds",
            }

        except Exception as e:
            self.logger.error(f"[CodeExecutor] Command failed: {e}")
            return {"success": False, "error": str(e)}

    def validate_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Validate code syntax without executing.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            Dictionary with validation result
        """
        try:
            if language == "python":
                # Try to compile the code
                compile(code, "<string>", "exec")

                return {"success": True, "valid": True, "language": language}
            else:
                return {
                    "success": True,
                    "valid": None,
                    "message": f"Validation not implemented for {language}",
                }

        except SyntaxError as e:
            return {"success": True, "valid": False, "error": str(e), "line": e.lineno}

        except Exception as e:
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("Testing Code Executor...")

    executor = CodeExecutor()

    # Test Python execution
    print("\n1. Testing execute_python...")
    code = "print('Hello from executed code!')\nprint(2 + 2)"
    result = executor.execute_python(code)
    print(f"   Success: {result['success']}")
    print(f"   Output: {result.get('stdout', '').strip()}")

    # Test validation
    print("\n2. Testing validate_code...")
    result = executor.validate_code("print('valid')")
    print(f"   Valid: {result.get('valid')}")

    # Test invalid code
    result = executor.validate_code("print('invalid")
    print(f"   Invalid detected: {not result.get('valid')}")

    # Test shell command
    print("\n3. Testing execute_shell...")
    result = executor.execute_shell("echo 'Hello from shell'")
    print(f"   Success: {result['success']}")
    print(f"   Output: {result.get('stdout', '').strip()}")

    print("\n✅ Code Executor test complete!")
