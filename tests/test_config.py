from pathlib import Path

import pytest

from src.config import DEFAULT_DATA_DIR, PROVIDER_DEFAULTS, load_settings


def test_load_settings_when_env_empty_uses_anthropic_defaults():
    settings = load_settings({})
    assert settings.llm_provider == "anthropic"
    assert settings.chat_model == PROVIDER_DEFAULTS["anthropic"].chat_model
    assert settings.data_dir == DEFAULT_DATA_DIR
    assert settings.data_dir.is_absolute()


def test_load_settings_when_overrides_set_uses_them(tmp_path):
    settings = load_settings(
        {"LLM_PROVIDER": "openai", "CHAT_MODEL": "chat-x", "MEMORY_MODEL": "mem-x", "DATA_DIR": str(tmp_path)}
    )
    assert (settings.llm_provider, settings.chat_model, settings.memory_model) == ("openai", "chat-x", "mem-x")
    assert settings.data_dir == Path(tmp_path).resolve()


def test_load_settings_when_blank_override_falls_back_to_provider_default():
    settings = load_settings({"LLM_PROVIDER": "openai", "CHAT_MODEL": ""})
    assert settings.chat_model == PROVIDER_DEFAULTS["openai"].chat_model


def test_load_settings_when_provider_unknown_raises_value_error():
    with pytest.raises(ValueError):
        load_settings({"LLM_PROVIDER": "unknown"})


def test_settings_paths_live_under_data_dir(tmp_path):
    settings = load_settings({"DATA_DIR": str(tmp_path)})
    assert settings.checkpoint_path.parent == settings.data_dir
    assert settings.vector_store_path.parent == settings.data_dir


def test_load_settings_when_ollama_uses_local_models_and_default_url():
    settings = load_settings({"LLM_PROVIDER": "ollama"})
    assert settings.chat_model == PROVIDER_DEFAULTS["ollama"].chat_model
    assert settings.provider.embedder_provider == "fastembed"
    assert settings.ollama_base_url == "http://localhost:11434"


def test_load_settings_when_ollama_url_set_uses_it():
    settings = load_settings({"LLM_PROVIDER": "ollama", "OLLAMA_BASE_URL": "http://gpu-box.test:11434"})
    assert settings.ollama_base_url == "http://gpu-box.test:11434"
