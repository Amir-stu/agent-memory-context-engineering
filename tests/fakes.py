"""Deterministic stand-ins for the two network/storage boundaries: the LLM and mem0."""

import itertools
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


class RecordingChatModel(BaseChatModel):
    """Chat model that records every prompt and answers via a pure function of it."""

    reply_fn: Callable[[Sequence[BaseMessage]], str]
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        reply = AIMessage(content=self.reply_fn(messages))
        return ChatResult(generations=[ChatGeneration(message=reply)])


class FakeMem0:
    """In-process imitation of mem0's Memory API (add/search/get_all/update/delete).

    With infer=True it stores each user utterance verbatim instead of asking an LLM
    to extract facts, so retrieval and scoping are testable without network calls.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self._ids = itertools.count(1)

    def add(self, messages, *, user_id=None, infer=True, metadata=None, **kwargs):
        texts = (
            [messages]
            if isinstance(messages, str)
            else [message["content"] for message in messages if message["role"] == "user" or not infer]
        )
        results = []
        for text in texts:
            memory_id = f"mem-{next(self._ids)}"
            self.rows[memory_id] = {
                "id": memory_id,
                "memory": text,
                "user_id": user_id,
                "updated_at": f"2026-01-01T00:00:{len(self.rows):02d}",
            }
            results.append({"id": memory_id, "memory": text, "event": "ADD"})
        return {"results": results}

    def _scoped(self, filters):
        return [row for row in self.rows.values() if row["user_id"] == filters["user_id"]]

    def search(self, query, *, top_k=20, filters=None, threshold=0.1, **kwargs):
        query_tokens = _tokens(query)
        scored = [
            {**row, "score": len(query_tokens & _tokens(row["memory"])) / max(len(query_tokens), 1)}
            for row in self._scoped(filters)
        ]
        hits = sorted((row for row in scored if row["score"] >= threshold), key=lambda row: row["score"], reverse=True)
        return {"results": hits[:top_k]}

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        return {"results": self._scoped(filters)[:top_k]}

    def update(self, memory_id, text=None, **kwargs):
        self.rows[memory_id] = {**self.rows[memory_id], "memory": text}
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        self.rows.pop(memory_id)
        return {"message": "Memory deleted successfully!"}


class ExplodingMem0(FakeMem0):
    """Backend whose writes fail, to prove background errors are surfaced."""

    def add(self, messages, **kwargs):
        raise RuntimeError("vector store unavailable")
