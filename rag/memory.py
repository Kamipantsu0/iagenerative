"""
memory.py
---------
Conversation memory for RAG Option 2.
Keeps a sliding window of past (user, assistant) turns to enable follow-up questions.
"""

from typing import List, Tuple


class ConversationMemory:
    """
    Stores the last `max_turns` conversation exchanges.

    Each turn is a (role, content) tuple where role is 'user' or 'assistant'.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._history: List[Tuple[str, str]] = []

    def add(self, role: str, content: str) -> None:
        """Append a message. Trims oldest turns when limit is exceeded."""
        self._history.append((role, content))
        # Keep only the last max_turns * 2 messages (each turn = 2 messages)
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]

    def get_history_text(self) -> str:
        """Return conversation history formatted for prompt injection."""
        if not self._history:
            return ""
        lines = []
        for role, content in self._history:
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    def get_history_list(self) -> List[dict]:
        """Return history as a list of dicts (serializable for session state)."""
        return [{"role": r, "content": c} for r, c in self._history]

    def load_from_list(self, history: List[dict]) -> None:
        """Restore memory from a serialized list of dicts."""
        self._history = [(h["role"], h["content"]) for h in history]

    def clear(self) -> None:
        """Reset conversation history."""
        self._history = []

    def __len__(self) -> int:
        return len(self._history) // 2  # number of full turns
