"""
Task Classifier - Maps natural language keywords to agent task types.

Single source of truth for keyword→task-type routing, used by both
the Orchestrator (classifying user requests) and the PlanningAgent
(assigning subtasks to agents).
"""

from typing import List, Tuple

# Ordered list of (task_type, keywords) pairs.
# Earlier entries take priority when multiple types match.
TASK_TYPE_KEYWORDS: List[Tuple[str, List[str]]] = [
    # Conversational / general — checked first so greetings and capability
    # questions never get routed to specialist agents.
    ("general",     ["hi ", "hi!", "hello", "hey ", "hey!", "how are you",
                     "what's up", "good morning", "good afternoon", "good evening",
                     "who are you", "what are you", "thank you", "thanks",
                     "bye", "goodbye",
                     "how well", "how good", "are you good", "are you able",
                     "what can you", "can you help", "do you know",
                     "tell me about yourself", "what do you do"]),
    ("planning",    ["plan", "organize", "break down", "steps"]),
    ("coding",      ["code", "script", "function", "program", "execute", "run"]),
    ("web",         ["search", "web", "internet", "lookup", "find online"]),
    ("data",        ["data", "dataset", "dataframe", "spreadsheet", "csv", "calculate",
                     "chart", "graph", "statistics", "aggregate", "correlation"]),
    ("automation",  ["automate", "control", "click", "type", "gui"]),
    ("security",    ["security", "virus", "malware", "suspicious", "threat",
                     "scan for", "hack", "infected", "intrusion"]),
    ("memory",      ["remember", "recall", "memory", "store", "retrieve"]),
    ("vision",      ["image", "picture", "photo", "visual", "see"]),
    ("audio",       ["audio", "sound", "voice", "speech", "listen"]),
    ("qa",          ["test", "verify", "qa", "quality"]),
    ("file",        ["file", "folder", "read", "write", "save", "load", "scan", "move"]),
]

# Short messages below this word count skip LLM classification and go
# straight to "general" — avoids a full inference call for one-liners.
SHORT_MESSAGE_WORD_THRESHOLD = 6

# Default when no keywords match — routes to GeneralAgent for open-ended/
# conversational requests that don't fit any specialist category.
DEFAULT_TASK_TYPE = "general"

# Word-count threshold above which a request is considered "long"
# and falls back to planning even without explicit planning keywords.
LONG_REQUEST_WORD_THRESHOLD = 10


def classify_task_type(text: str) -> str:
    """
    Classify a natural-language string into a task type.

    Iterates TASK_TYPE_KEYWORDS in priority order and returns the
    first match. Falls back to DEFAULT_TASK_TYPE if nothing matches.

    Args:
        text: Natural language text to classify.

    Returns:
        Task type string (e.g. "file", "coding", "web").
    """
    text_lower = text.lower()
    for task_type, keywords in TASK_TYPE_KEYWORDS:
        if any(kw in text_lower for kw in keywords):
            return task_type
    return DEFAULT_TASK_TYPE
