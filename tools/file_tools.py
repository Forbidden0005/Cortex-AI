"""
File Tools - File system operations for agents

Provides safe, logged file operations that agents can use.
"""

import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path
from pathlib import Path as PathLib
from typing import Any, Dict, Optional

sys.path.append(str(PathLib(__file__).parent.parent))
from core.logger import get_logger


class FileTools:
    """
    File system operation tools.

    All operations are logged and validated for safety.
    """

    def __init__(self):
        self.logger = get_logger()

    def read_file(self, filepath: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Read file contents.

        Args:
            filepath: Path to file
            encoding: File encoding

        Returns:
            Dictionary with content and metadata
        """
        try:
            path = Path(filepath)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            with open(path, "r", encoding=encoding) as f:
                content = f.read()

            self.logger.info(
                f"[FileTools] Read file: {filepath} ({len(content)} chars)"
            )

            return {
                "success": True,
                "filepath": str(path.absolute()),
                "content": content,
                "size": len(content),
                "lines": len(content.splitlines()),
                "encoding": encoding,
            }

        except Exception as e:
            self.logger.error(f"[FileTools] Read failed: {filepath} - {e}")
            return {"success": False, "filepath": filepath, "error": str(e)}

    def write_file(
        self, filepath: str, content: str, encoding: str = "utf-8", mode: str = "w"
    ) -> Dict[str, Any]:
        """
        Write content to file.

        Args:
            filepath: Path to file
            content: Content to write
            encoding: File encoding
            mode: Write mode ('w' for overwrite, 'a' for append)

        Returns:
            Dictionary with result
        """
        try:
            path = Path(filepath)

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, mode, encoding=encoding) as f:
                f.write(content)

            self.logger.info(
                f"[FileTools] Wrote file: {filepath} ({len(content)} chars)"
            )

            return {
                "success": True,
                "filepath": str(path.absolute()),
                "bytes_written": len(content.encode(encoding)),
                "mode": mode,
            }

        except Exception as e:
            self.logger.error(f"[FileTools] Write failed: {filepath} - {e}")
            return {"success": False, "filepath": filepath, "error": str(e)}

    def list_directory(
        self, directory: str, pattern: Optional[str] = None, recursive: bool = False
    ) -> Dict[str, Any]:
        """
        List directory contents.

        Args:
            directory: Directory path
            pattern: Optional glob pattern (e.g., "*.py")
            recursive: Recursively list subdirectories

        Returns:
            Dictionary with file list
        """
        try:
            path = Path(directory)

            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")

            if not path.is_dir():
                raise NotADirectoryError(f"Not a directory: {directory}")

            # Get files
            if pattern:
                if recursive:
                    files = list(path.rglob(pattern))
                else:
                    files = list(path.glob(pattern))
            else:
                if recursive:
                    files = list(path.rglob("*"))
                else:
                    files = list(path.glob("*"))

            # Organize results
            result = {"files": [], "directories": [], "total": 0}

            for item in files:
                item_info = {
                    "name": item.name,
                    "path": str(item.absolute()),
                    "size": item.stat().st_size if item.is_file() else 0,
                    "modified": datetime.fromtimestamp(
                        item.stat().st_mtime
                    ).isoformat(),
                }

                if item.is_file():
                    result["files"].append(item_info)
                elif item.is_dir():
                    result["directories"].append(item_info)

                result["total"] += 1

            self.logger.info(
                f"[FileTools] Listed directory: {directory} ({result['total']} items)"
            )

            return {"success": True, "directory": str(path.absolute()), **result}

        except Exception as e:
            self.logger.error(f"[FileTools] List failed: {directory} - {e}")
            return {"success": False, "directory": directory, "error": str(e)}

    def copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy file.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            Dictionary with result
        """
        try:
            src = Path(source)
            dst = Path(destination)

            if not src.exists():
                raise FileNotFoundError(f"Source not found: {source}")

            # Create destination directory if needed
            dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src, dst)

            self.logger.info(f"[FileTools] Copied: {source} -> {destination}")

            return {
                "success": True,
                "source": str(src.absolute()),
                "destination": str(dst.absolute()),
                "size": dst.stat().st_size,
            }

        except Exception as e:
            self.logger.error(
                f"[FileTools] Copy failed: {source} -> {destination} - {e}"
            )
            return {
                "success": False,
                "source": source,
                "destination": destination,
                "error": str(e),
            }

    def move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Move/rename file.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            Dictionary with result
        """
        try:
            src = Path(source)
            dst = Path(destination)

            if not src.exists():
                raise FileNotFoundError(f"Source not found: {source}")

            # Create destination directory if needed
            dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(src, dst)

            self.logger.info(f"[FileTools] Moved: {source} -> {destination}")

            return {
                "success": True,
                "source": source,
                "destination": str(dst.absolute()),
            }

        except Exception as e:
            self.logger.error(
                f"[FileTools] Move failed: {source} -> {destination} - {e}"
            )
            return {
                "success": False,
                "source": source,
                "destination": destination,
                "error": str(e),
            }

    def delete_file(self, filepath: str) -> Dict[str, Any]:
        """
        Delete file.

        Args:
            filepath: Path to file

        Returns:
            Dictionary with result
        """
        try:
            path = Path(filepath)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            path.unlink()

            self.logger.warning(f"[FileTools] Deleted: {filepath}")

            return {"success": True, "filepath": filepath, "deleted": True}

        except Exception as e:
            self.logger.error(f"[FileTools] Delete failed: {filepath} - {e}")
            return {"success": False, "filepath": filepath, "error": str(e)}

    def file_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get file information.

        Args:
            filepath: Path to file

        Returns:
            Dictionary with file metadata
        """
        try:
            path = Path(filepath)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            stat = path.stat()

            # Calculate file hash for integrity
            hash_md5 = hashlib.md5()
            if path.is_file():
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)

            return {
                "success": True,
                "filepath": str(path.absolute()),
                "name": path.name,
                "extension": path.suffix,
                "size": stat.st_size,
                "is_file": path.is_file(),
                "is_directory": path.is_dir(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "md5": hash_md5.hexdigest() if path.is_file() else None,
            }

        except Exception as e:
            self.logger.error(f"[FileTools] Info failed: {filepath} - {e}")
            return {"success": False, "filepath": filepath, "error": str(e)}

    def create_directory(self, directory: str) -> Dict[str, Any]:
        """
        Create directory.

        Args:
            directory: Directory path

        Returns:
            Dictionary with result
        """
        try:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"[FileTools] Created directory: {directory}")

            return {"success": True, "directory": str(path.absolute()), "created": True}

        except Exception as e:
            self.logger.error(f"[FileTools] Create directory failed: {directory} - {e}")
            return {"success": False, "directory": directory, "error": str(e)}

    def search_files(
        self, directory: str, query: str, case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """
        Search for text in files.

        Args:
            directory: Directory to search
            query: Text to search for
            case_sensitive: Whether search is case-sensitive

        Returns:
            Dictionary with search results
        """
        try:
            path = Path(directory)

            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")

            results = []

            # Search in text files
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            content = f.read()

                        # Search
                        search_content = content if case_sensitive else content.lower()
                        search_query = query if case_sensitive else query.lower()

                        if search_query in search_content:
                            # Find line numbers
                            lines = content.splitlines()
                            matches = []

                            for i, line in enumerate(lines, 1):
                                search_line = line if case_sensitive else line.lower()
                                if search_query in search_line:
                                    matches.append({"line": i, "content": line.strip()})

                            results.append(
                                {
                                    "file": str(file_path.absolute()),
                                    "matches": len(matches),
                                    "lines": matches[:10],  # First 10 matches
                                }
                            )

                    except (OSError, PermissionError):
                        # Skip files that can't be read
                        pass

            self.logger.info(
                f"[FileTools] Searched {directory} for '{query}' - {len(results)} files"
            )

            return {
                "success": True,
                "directory": str(path.absolute()),
                "query": query,
                "files_found": len(results),
                "results": results,
            }

        except Exception as e:
            self.logger.error(f"[FileTools] Search failed: {directory} - {e}")
            return {"success": False, "directory": directory, "error": str(e)}


if __name__ == "__main__":
    # Test file tools
    print("Testing File Tools...")

    tools = FileTools()

    # Test write
    print("\n1. Testing write_file...")
    result = tools.write_file("test_file.txt", "Hello, World!")
    print(f"   Success: {result['success']}")

    # Test read
    print("\n2. Testing read_file...")
    result = tools.read_file("test_file.txt")
    print(f"   Success: {result['success']}")
    print(f"   Content: {result.get('content', 'N/A')}")

    # Test list
    print("\n3. Testing list_directory...")
    result = tools.list_directory(".")
    print(f"   Success: {result['success']}")
    print(f"   Total items: {result.get('total', 0)}")

    # Test file info
    print("\n4. Testing file_info...")
    result = tools.file_info("test_file.txt")
    print(f"   Success: {result['success']}")
    print(f"   Size: {result.get('size', 0)} bytes")

    # Test delete
    print("\n5. Testing delete_file...")
    result = tools.delete_file("test_file.txt")
    print(f"   Success: {result['success']}")

    print("\n✅ File Tools test complete!")
