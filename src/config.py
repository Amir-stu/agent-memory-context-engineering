"""Runtime settings, read once from the environment. Every tunable lives here."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class ProviderDefaults:
    chat_model: str
    memory_model: str
    embedder_provider: str
    embedding_model: str
    embedding_dims: int


PROVIDER_DEFAULTS: Mapping[str, ProviderDefaults] = {
    # Anthropic has no embeddings API, so embeddings run locally (ONNX, no torch).
    "anthropic": ProviderDefaults(
        chat_model="claude-sonnet-5-5",
        memory_model="claude-haiku-5-5",
        embedder_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dims=384,
    ),
    "openai": ProviderDefaults(
        chat_model="gpt-5-mini",
        memory_model="gpt-5-mini",
        embedder_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dims=1536,
    ),
    # Fully local: no API key. Chat and fact extraction run on an Ollama model.
    "ollama": ProviderDefaults(
        chat_model="qwen2.5:7b",
        memory_model="qwen2.5:7b",
        embedder_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dims=384,
    ),
}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = DEFAULT_DATA_DIR
    llm_provider: str = "anthropic"
    chat_model: str = PROVIDER_DEFAULTS["anthropic"].chat_model
    memory_model: str = PROVIDER_DEFAULTS["anthropic"].memory_model
    retrieval_top_k: int = 8
    history_window: int = 12
    max_memory_chars: int = 2000
    max_response_tokens: int = 1024
    memory_collection: str = "companion_memories"
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL

    @property
    def provider(self) -> ProviderDefaults:
        return PROVIDER_DEFAULTS[self.llm_provider]

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.db"

    @property
    def vector_store_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def memory_history_path(self) -> Path:
        return self.data_dir / "mem0_history.db"


def load_settings(env: Mapping[str, str] = os.environ) -> Settings:
    provider_name = env.get("LLM_PROVIDER") or "anthropic"
    if provider_name not in PROVIDER_DEFAULTS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDER_DEFAULTS)}, got {provider_name!r}")
    defaults = PROVIDER_DEFAULTS[provider_name]
    data_dir = Path(env.get("DATA_DIR") or DEFAULT_DATA_DIR).resolve()
    return Settings(
        data_dir=data_dir,
        llm_provider=provider_name,
        chat_model=env.get("CHAT_MODEL") or defaults.chat_model,
        memory_model=env.get("MEMORY_MODEL") or defaults.memory_model,
        ollama_base_url=env.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL,
    )
