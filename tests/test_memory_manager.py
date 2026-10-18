import pytest

from src.config import Settings
from src.memory_manager import MemoryManager, MemoryRecord, build_mem0_config, validate_identifier
from tests.fakes import ExplodingMem0

USER = "user-zed"
OTHER_USER = "user-quux"


def test_validate_identifier_when_valid_returns_stripped_id():
    assert validate_identifier("  user-zed_01 ", "user_id") == "user-zed_01"


@pytest.mark.parametrize("bad_id", ["", "   ", "has space", "../etc", "x" * 65, "semi;colon"])
def test_validate_identifier_when_malformed_raises_value_error(bad_id):
    with pytest.raises(ValueError, match="user_id"):
        validate_identifier(bad_id, "user_id")


def test_search_when_memories_exist_returns_records_scoped_to_user(memory_manager):
    memory_manager.add_fact(USER, "Prefers DuckDB for analytics")
    memory_manager.add_fact(OTHER_USER, "Prefers DuckDB for analytics too")

    results = memory_manager.search(USER, "which database for analytics")

    assert [result.text for result in results] == ["Prefers DuckDB for analytics"]
    assert isinstance(results[0], MemoryRecord)
    assert results[0].score > 0


def test_search_when_query_blank_returns_empty_without_backend_call(memory_manager):
    memory_manager.add_fact(USER, "Name is Zed")
    assert memory_manager.search(USER, "   ") == ()


def test_remember_when_turn_given_stores_user_utterance_and_reports_events(memory_manager):
    events = memory_manager.remember(
        USER, [{"role": "user", "content": "I am Zed"}, {"role": "assistant", "content": "Hi"}]
    )
    assert [(event.event, event.text) for event in events] == [("ADD", "I am Zed")]
    assert [record.text for record in memory_manager.list_memories(USER)] == ["I am Zed"]


def test_remember_async_when_awaited_persists_and_reports_no_errors(memory_manager):
    memory_manager.remember_async(USER, [{"role": "user", "content": "Goal: ML platform role"}])
    assert memory_manager.wait_for_pending() == ()
    assert len(memory_manager.list_memories(USER)) == 1


def test_remember_async_when_backend_fails_surfaces_error_on_wait():
    manager = MemoryManager(ExplodingMem0())
    manager.remember_async(USER, [{"role": "user", "content": "hello"}])
    errors = manager.wait_for_pending()
    manager.close()
    assert len(errors) == 1
    assert "vector store unavailable" in str(errors[0])


def test_wait_for_pending_when_called_twice_does_not_repeat_errors():
    manager = MemoryManager(ExplodingMem0())
    manager.remember_async(USER, [{"role": "user", "content": "hello"}])
    manager.wait_for_pending()
    assert manager.wait_for_pending() == ()
    manager.close()


def test_edit_when_called_changes_memory_text(memory_manager):
    record = memory_manager.add_fact(USER, "Uses PostgreSQL")
    memory_manager.edit(record.id, "Uses DuckDB")
    assert [item.text for item in memory_manager.list_memories(USER)] == ["Uses DuckDB"]


def test_edit_when_text_blank_raises_value_error(memory_manager):
    record = memory_manager.add_fact(USER, "Uses PostgreSQL")
    with pytest.raises(ValueError):
        memory_manager.edit(record.id, "  ")


def test_forget_when_called_removes_only_that_memory(memory_manager):
    kept = memory_manager.add_fact(USER, "Name is Zed")
    dropped = memory_manager.add_fact(USER, "Likes tea")
    memory_manager.forget(dropped.id)
    assert [item.id for item in memory_manager.list_memories(USER)] == [kept.id]


def test_add_fact_when_text_blank_raises_value_error(memory_manager):
    with pytest.raises(ValueError):
        memory_manager.add_fact(USER, "")


def test_build_mem0_config_when_anthropic_uses_local_embedder_and_data_dir(tmp_path):
    config = build_mem0_config(Settings(data_dir=tmp_path))
    assert config["llm"]["provider"] == "anthropic"
    assert config["embedder"]["provider"] == "fastembed"
    assert config["vector_store"]["config"]["path"] == str(tmp_path / "qdrant")
    assert config["vector_store"]["config"]["embedding_model_dims"] == 384
    assert config["history_db_path"] == str(tmp_path / "mem0_history.db")


def test_build_mem0_config_when_openai_uses_openai_embedder(tmp_path):
    config = build_mem0_config(Settings(data_dir=tmp_path, llm_provider="openai"))
    assert config["llm"]["provider"] == "openai"
    assert config["embedder"]["provider"] == "openai"
    assert config["vector_store"]["config"]["embedding_model_dims"] == 1536


def test_build_mem0_config_when_ollama_points_llm_at_local_server(tmp_path):
    config = build_mem0_config(Settings(data_dir=tmp_path, llm_provider="ollama", memory_model="qwen2.5:7b"))
    assert config["llm"]["provider"] == "ollama"
    assert config["llm"]["config"]["ollama_base_url"] == "http://localhost:11434"
    assert config["llm"]["config"]["model"] == "qwen2.5:7b"
    assert config["embedder"]["provider"] == "fastembed"


def test_build_mem0_config_when_hosted_provider_omits_ollama_url(tmp_path):
    config = build_mem0_config(Settings(data_dir=tmp_path))
    assert "ollama_base_url" not in config["llm"]["config"]
