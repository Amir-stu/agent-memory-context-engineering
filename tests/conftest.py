import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.agent_graph import CompanionAgent
from src.config import Settings
from src.memory_manager import MemoryManager
from tests.fakes import FakeMem0, RecordingChatModel

MEMORY_SECTION_START = "<user_memory>"
MEMORY_SECTION_END = "</user_memory>"


def echo_memory_section(messages) -> str:
    """Answer with whatever the context engine injected, so tests can see it."""
    system_text = messages[0].content
    start = system_text.index(MEMORY_SECTION_START)
    end = system_text.index(MEMORY_SECTION_END)
    return system_text[start : end + len(MEMORY_SECTION_END)]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path)


@pytest.fixture
def fake_backend() -> FakeMem0:
    return FakeMem0()


@pytest.fixture
def memory_manager(fake_backend, settings):
    manager = MemoryManager(fake_backend, top_k=settings.retrieval_top_k)
    yield manager
    manager.close()


@pytest.fixture
def chat_model() -> RecordingChatModel:
    return RecordingChatModel(reply_fn=echo_memory_section, calls=[])


@pytest.fixture
def agent(chat_model, memory_manager, settings) -> CompanionAgent:
    return CompanionAgent(chat_model, memory_manager, InMemorySaver(), settings)
