"""
Session Store - Persists loop execution state to disk between runs.

Writes a JSON checkpoint after each task completes or fails, so a crash
or interruption mid-queue does not lose all progress. On restart the
orchestrator can inspect the last checkpoint to understand where the
loop was and what failed.

Layout on disk:
    sessions/
      <session_id>/
        checkpoint.json   -- latest snapshot (overwritten each save)
        events.jsonl      -- append-only event log (one JSON object per line)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_logger = logging.getLogger("Cortex")

# Default storage directory anchored to the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SESSIONS_DIR = _PROJECT_ROOT / "sessions"


class SessionStore:
    """
    Saves and loads loop execution state as JSON checkpoints.

    Each session gets its own subdirectory under sessions_dir. The
    checkpoint file is always overwritten with the latest state;
    the events file is append-only and accumulates the full history.
    """

    def __init__(self, sessions_dir: str = ""):
        base = Path(sessions_dir) if sessions_dir else _DEFAULT_SESSIONS_DIR
        self.sessions_dir = base
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        session_id: str,
        pending_count: int,
        completed_count: int,
        failed_count: int,
        frozen_tasks: List[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write a snapshot of the current loop state.

        Args:
            session_id:      Unique identifier for this execution session.
            pending_count:   Tasks still waiting to run.
            completed_count: Tasks that finished successfully.
            failed_count:    Tasks permanently failed or frozen.
            frozen_tasks:    task_ids frozen due to churn.
            extra:           Optional additional metadata to include.
        """
        session_dir = self._session_dir(session_id)

        checkpoint = {
            "session_id": session_id,
            "updated_at": datetime.now().isoformat(),
            "pending": pending_count,
            "completed": completed_count,
            "failed": failed_count,
            "frozen_tasks": frozen_tasks,
            **(extra or {}),
        }

        try:
            with open(session_dir / "checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2)
        except OSError as e:
            _logger.error(
                f"[SessionStore] Failed to save checkpoint for {session_id}: {e}"
            )

    def load_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a previously saved checkpoint.

        Args:
            session_id: Session to load.

        Returns:
            Checkpoint dict, or None if not found or unreadable.
        """
        path = self.sessions_dir / session_id / "checkpoint.json"
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            _logger.error(
                f"[SessionStore] Failed to load checkpoint for {session_id}: {e}"
            )
            return None

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def log_event(
        self,
        session_id: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Append a single event to the session's append-only event log.

        Args:
            session_id:  Session identifier.
            event_type:  Short label (e.g. "task_completed", "loop_started").
            data:        Event payload (must be JSON-serialisable).
        """
        session_dir = self._session_dir(session_id)
        entry = {
            "ts": datetime.now().isoformat(),
            "event": event_type,
            **data,
        }
        try:
            with open(session_dir / "events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            _logger.error(
                f"[SessionStore] Failed to write event for {session_id}: {e}"
            )

    # ------------------------------------------------------------------
    # Session discovery
    # ------------------------------------------------------------------

    def list_sessions(self) -> List[str]:
        """Return all session IDs that have a saved checkpoint."""
        if not self.sessions_dir.exists():
            return []
        return [
            d.name
            for d in self.sessions_dir.iterdir()
            if d.is_dir() and (d / "checkpoint.json").exists()
        ]

    def get_latest_session(self) -> Optional[str]:
        """
        Return the session_id of the most recently updated checkpoint,
        or None if no sessions exist.
        """
        sessions = self.list_sessions()
        if not sessions:
            return None

        def _updated_at(sid: str) -> str:
            cp = self.load_checkpoint(sid)
            return cp.get("updated_at", "") if cp else ""

        return max(sessions, key=_updated_at)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_dir(self, session_id: str) -> Path:
        """Return (and create) the directory for a session."""
        path = self.sessions_dir / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path
