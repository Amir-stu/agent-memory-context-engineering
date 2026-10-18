"""LangGraph state machine: retrieve memory -> build context -> respond -> persist memory.

Two memory layers with different lifetimes:
  * short-term: the thread's message list, checkpointed per (user, thread) by a LangGraph saver;
  * long-term: semantic facts in mem0, scoped per user and shared by all of that user's threads.
Clearing a thread deletes its checkpoint only; the user's long-term memory is untouched.
"""

import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.config import Settings
from src.context_builder import build_context, build_system_prompt, latest_user_text
from src.memory_manager import MemoryManager, MemoryRecord, create_memory_manager, validate_identifier

THREAD_SEPARATOR = "::"


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    memories: list[dict]


@dataclass(frozen=True)
class AgentTurn:
    answer: str
    memories: tuple[MemoryRecord, ...]


def scoped_thread_id(user_id: str, thread_id: str) -> str:
    """Namespace checkpoints by user so two users' 'main' threads never collide."""
    return f"{user_id}{THREAD_SEPARATOR}{thread_id}"


def build_graph(
    chat_model: BaseChatModel,
    memory: MemoryManager,
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
    clock: Callable[[], date] = date.today,
):
    def retrieve_memories(state: AgentState) -> dict:
        records = memory.search(state["user_id"], latest_user_text(state["messages"]))
        return {"memories": [record.model_dump() for record in records]}

    def generate_response(state: AgentState) -> dict:
        records = [MemoryRecord(**row) for row in state["memories"]]
        system_prompt = build_system_prompt(records, today=clock(), max_memory_chars=settings.max_memory_chars)
        context = build_context(system_prompt, state["messages"], window=settings.history_window)
        return {"messages": [chat_model.invoke(context)]}

    def persist_memories(state: AgentState) -> dict:
        turn = [
            {"role": "user", "content": latest_user_text(state["messages"])},
            {"role": "assistant", "content": state["messages"][-1].text},
        ]
        memory.remember_async(state["user_id"], turn)
        return {}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve_memories", retrieve_memories)
    graph.add_node("generate_response", generate_response)
    graph.add_node("persist_memories", persist_memories)
    graph.add_edge(START, "retrieve_memories")
    graph.add_edge("retrieve_memories", "generate_response")
    graph.add_edge("generate_response", "persist_memories")
    graph.add_edge("persist_memories", END)
    return graph.compile(checkpointer=checkpointer)


class CompanionAgent:
    def __init__(
        self,
        chat_model: BaseChatModel,
        memory: MemoryManager,
        checkpointer: BaseCheckpointSaver,
        settings: Settings,
    ) -> None:
        self.memory = memory
        self._checkpointer = checkpointer
        self._graph = build_graph(chat_model, memory, settings, checkpointer)

    @staticmethod
    def _config(user_id: str, thread_id: str) -> dict:
        scoped = scoped_thread_id(validate_identifier(user_id, "user_id"), validate_identifier(thread_id, "thread_id"))
        return {"configurable": {"thread_id": scoped}}

    def reply(self, user_id: str, thread_id: str, text: str) -> AgentTurn:
        config = self._config(user_id, thread_id)
        question = HumanMessage(content=text, id=str(uuid.uuid4()))
        try:
            state = self._graph.invoke(
                {"messages": [question], "user_id": validate_identifier(user_id, "user_id")}, config
            )
        except Exception:
            # The checkpoint already holds the question; drop it so the thread has no unanswered turn.
            self._graph.update_state(config, {"messages": [RemoveMessage(id=question.id)]})
            raise
        return AgentTurn(
            answer=state["messages"][-1].text,
            memories=tuple(MemoryRecord(**row) for row in state["memories"]),
        )

    def history(self, user_id: str, thread_id: str) -> list[BaseMessage]:
        snapshot = self._graph.get_state(self._config(user_id, thread_id))
        return list(snapshot.values.get("messages", []))

    def clear_thread(self, user_id: str, thread_id: str) -> None:
        self._checkpointer.delete_thread(self._config(user_id, thread_id)["configurable"]["thread_id"])

    def list_threads(self, user_id: str) -> tuple[str, ...]:
        prefix = scoped_thread_id(validate_identifier(user_id, "user_id"), "")
        thread_ids = {checkpoint.config["configurable"]["thread_id"] for checkpoint in self._checkpointer.list(None)}
        return tuple(sorted(thread_id[len(prefix) :] for thread_id in thread_ids if thread_id.startswith(prefix)))

    def close(self) -> None:
        self.memory.close()
        connection = getattr(self._checkpointer, "conn", None)
        if connection is not None:
            connection.close()


def create_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.chat_model, base_url=settings.ollama_base_url, num_predict=settings.max_response_tokens
        )
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.chat_model, max_tokens=settings.max_response_tokens)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=settings.chat_model, max_tokens=settings.max_response_tokens)


def open_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # Streamlit reruns scripts on different threads; SqliteSaver serialises access itself.
    return SqliteSaver(sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False))


def create_agent(settings: Settings) -> CompanionAgent:
    return CompanionAgent(
        create_chat_model(settings), create_memory_manager(settings), open_checkpointer(settings), settings
    )
