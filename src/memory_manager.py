"""Long-term semantic memory: a thin, typed layer over mem0 scoped by user_id.

mem0 does the hard part on write: an LLM extracts candidate facts from a turn, embeds
them, finds similar existing memories and decides ADD / UPDATE / DELETE / NONE. That
reconciliation step is what keeps a corrected preference from sitting next to the stale one.
"""

import logging
import os
import re
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from src.config import Settings

logger = logging.getLogger(__name__)

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LIST_LIMIT = 200


class MemoryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    score: float | None = None
    updated_at: str | None = None


class MemoryEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    event: str


class MemoryBackend(Protocol):
    """The subset of mem0's `Memory` API this project depends on."""

    def add(self, messages: Any, *, user_id: str | None = None, infer: bool = True, **kwargs: Any) -> dict: ...
    def search(self, query: str, *, top_k: int = 20, filters: dict | None = None, **kwargs: Any) -> dict: ...
    def get_all(self, *, filters: dict | None = None, top_k: int = 20, **kwargs: Any) -> dict: ...
    def update(self, memory_id: str, text: str | None = None, **kwargs: Any) -> dict: ...
    def delete(self, memory_id: str) -> Any: ...


def validate_identifier(value: str, label: str) -> str:
    cleaned = value.strip()
    if not IDENTIFIER_PATTERN.fullmatch(cleaned):
        raise ValueError(f"{label} must be 1-64 characters of letters, digits, '-' or '_'")
    return cleaned


def _require_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("memory text must not be blank")
    return cleaned


def _to_record(row: dict) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        text=row["memory"],
        score=row.get("score"),
        updated_at=row.get("updated_at") or row.get("created_at"),
    )


class MemoryManager:
    def __init__(self, backend: MemoryBackend, *, top_k: int = 8) -> None:
        self._backend = backend
        self._top_k = top_k
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-writer")
        self._pending: tuple[Future, ...] = ()
        self._lock = threading.Lock()

    def search(self, user_id: str, query: str) -> tuple[MemoryRecord, ...]:
        if not query.strip():
            return ()
        response = self._backend.search(
            query, top_k=self._top_k, filters={"user_id": validate_identifier(user_id, "user_id")}
        )
        return tuple(_to_record(row) for row in response["results"])

    def list_memories(self, user_id: str) -> tuple[MemoryRecord, ...]:
        response = self._backend.get_all(filters={"user_id": validate_identifier(user_id, "user_id")}, top_k=LIST_LIMIT)
        return tuple(_to_record(row) for row in response["results"])

    def remember(self, user_id: str, messages: Sequence[dict]) -> tuple[MemoryEvent, ...]:
        response = self._backend.add(list(messages), user_id=validate_identifier(user_id, "user_id"), infer=True)
        return tuple(
            MemoryEvent(id=row["id"], text=row["memory"], event=row.get("event", "ADD")) for row in response["results"]
        )

    def remember_async(self, user_id: str, messages: Sequence[dict]) -> Future:
        """Persist a turn off the response path; errors surface via `wait_for_pending`."""
        future = self._executor.submit(self.remember, user_id, tuple(messages))
        with self._lock:
            self._pending = (*self._pending, future)
        return future

    def wait_for_pending(self) -> tuple[BaseException, ...]:
        with self._lock:
            pending, self._pending = self._pending, ()
        errors = tuple(error for error in (future.exception() for future in pending) if error)
        for error in errors:
            logger.error("background memory write failed: %s", error)
        return errors

    def add_fact(self, user_id: str, text: str) -> MemoryRecord:
        response = self._backend.add(_require_text(text), user_id=validate_identifier(user_id, "user_id"), infer=False)
        return _to_record(response["results"][0])

    def edit(self, memory_id: str, text: str) -> None:
        self._backend.update(memory_id, text=_require_text(text))

    def forget(self, memory_id: str) -> None:
        self._backend.delete(memory_id)

    def close(self) -> None:
        self.wait_for_pending()
        self._executor.shutdown(wait=True)


def build_mem0_config(settings: Settings) -> dict:
    provider = settings.provider
    llm_config: dict[str, Any] = {"model": settings.memory_model, "temperature": 0.0}
    if settings.llm_provider == "ollama":
        llm_config = {**llm_config, "ollama_base_url": settings.ollama_base_url}
    return {
        "llm": {"provider": settings.llm_provider, "config": llm_config},
        "embedder": {
            "provider": provider.embedder_provider,
            "config": {"model": provider.embedding_model, "embedding_dims": provider.embedding_dims},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": settings.memory_collection,
                "path": str(settings.vector_store_path),
                "on_disk": True,
                "embedding_model_dims": provider.embedding_dims,
            },
        },
        "history_db_path": str(settings.memory_history_path),
    }


def create_memory_manager(settings: Settings) -> MemoryManager:
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    from mem0 import Memory

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    backend = Memory.from_config(build_mem0_config(settings))
    return MemoryManager(backend, top_k=settings.retrieval_top_k)
