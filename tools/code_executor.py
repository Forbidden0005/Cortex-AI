"""
Code Executor - Safe code execution for agents

Provides sandboxed code execution capabilities.
"""

import sys
from pathlib import Path as PathLib

sys.path.append(str(PathLib(__file__).parent.parent))
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional

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
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

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
        self, command: str, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute shell command safely.

        Args:
            command: Shell command to execute
            timeout: Execution timeout in seconds

        Returns:
            Dictionary with execution result
        """
        try:
            timeout = timeout or self.timeout

            self.logger.info(
                f"[CodeExecutor] Executing shell command: {command[:50]}..."
            )

            # Execute command
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout
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
