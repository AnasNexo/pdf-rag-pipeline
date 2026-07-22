"""Conversational memory for multi-turn chat.

The base Q&A path treats every question independently, so follow-ups that
depend on earlier turns ("what about its revenue?", "and who wrote it?") have
no referent. ConversationMemory keeps a bounded window of recent question/
answer pairs and exposes it two ways:

  - as a transcript injected into the generation prompt, so the LLM can
    resolve pronouns and elliptical references against what was already said;
  - as context for query rewriting (rag.query_transform.rewrite_with_history),
    so retrieval searches for a standalone version of the follow-up rather
    than the literal, context-free wording.

Only turns are stored, never the retrieved chunks — history is about the
conversation, not the corpus.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from config import MEMORY_MAX_TURNS


@dataclass
class Turn:
    """One completed exchange in the conversation.

    Attributes:
        question (str): What the user asked.
        answer (str): The assistant's answer.
    """

    question: str
    answer: str


class ConversationMemory:
    """A bounded, in-memory window of recent conversation turns.

    Holds at most ``max_turns`` exchanges; older turns are evicted as new ones
    arrive. Lives for the duration of one chat session and is not persisted.
    """

    def __init__(self, max_turns: int = MEMORY_MAX_TURNS) -> None:
        """Create an empty memory.

        Args:
            max_turns (int): Maximum number of turns to retain.
        """
        self.max_turns = max_turns
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, question: str, answer: str) -> None:
        """Record a completed turn, evicting the oldest if at capacity.

        Args:
            question (str): The user's question.
            answer (str): The assistant's answer.
        """
        self._turns.append(Turn(question=question, answer=answer))

    def clear(self) -> None:
        """Forget all stored turns."""
        self._turns.clear()

    def is_empty(self) -> bool:
        """Whether any turns have been recorded.

        Returns:
            bool: True if there is no history yet.
        """
        return not self._turns

    def as_transcript(self, max_answer_chars: int = 600) -> str:
        """Render recent turns as a plain-text transcript for a prompt.

        Answers are truncated so a long earlier answer cannot crowd out the
        retrieved document context in the generation prompt.

        Args:
            max_answer_chars (int): Per-answer truncation length.

        Returns:
            str: "User: ...\\nAssistant: ..." lines, oldest first; "" if empty.
        """
        lines: list[str] = []
        for turn in self._turns:
            answer = turn.answer.strip()
            if len(answer) > max_answer_chars:
                answer = answer[:max_answer_chars].rstrip() + "…"
            lines.append(f"User: {turn.question.strip()}")
            lines.append(f"Assistant: {answer}")
        return "\n".join(lines)
