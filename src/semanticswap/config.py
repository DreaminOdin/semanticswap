"""Zentrale Konfiguration (config.yaml), siehe PAD Abschnitt 4."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    host: str = "127.0.0.1"  # Server-Betrieb: "0.0.0.0" (nur mit Auth!)
    port: int = 8080
    api_style: str = "openai"
    # Öffentliches Pfad-Präfix, wenn die App hinter einem Reverse-Proxy unter
    # einem Unterpfad läuft (z. B. "/semanticswap" auf apimanufaktur.de).
    # Leer lassen für direkten Betrieb.
    root_path: str = ""
    # Basic Auth für GUI + API; leer = offen (nur für localhost-Betrieb!).
    # Überschreibbar per Env: SEMANTICSWAP_AUTH_USER / SEMANTICSWAP_AUTH_PASSWORD
    auth_user: str | None = None
    auth_password: str | None = None
    # Auth v2 (docs/plan-auth-v2.md): Geräte-Cookie-Login + Tailnet-Vertrauen
    device_cookie_days: int = 90
    trust_tailnet: bool = False           # Tailnet-Geräte ohne Passwort ...
    tailnet_users: list[str] = Field(default_factory=list)  # ... nur diese Accounts


class MainLLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "ollama/gemma4:26b"
    api_base: str | None = None  # z. B. "http://gx10:11434" für Remote-Ollama
    api_key: str | None = None   # z. B. LiteLLM-Proxy master_key
    # Clients dürfen das Modell pro Request wählen (Dropdown im Studio-Chat).
    # client_model_prefix wird dem Client-Modellnamen vorangestellt
    # (LiteLLM-Provider-Prefix, z. B. "openai/" für einen LiteLLM-Proxy).
    allow_client_model: bool = False
    client_model_prefix: str = ""
    max_context_tokens: int = 8192
    trigger_thresholds: list[float] = Field(default_factory=lambda: [0.5, 0.9])


class SubAgentTasks(BaseModel):
    summarization: str = "ollama/llama3:8b-instruct-q5_K_M"
    entity_extraction: str = "ollama/mistral:7b-instruct"
    synthesizer: str = "ollama/llama3:8b-instruct-q5_K_M"


class SubAgentsConfig(BaseModel):
    provider: str = "ollama"
    concurrency_limit: int = 4
    processing_mode: str = "batch"
    api_base: str | None = None  # Sub-Agenten können auf anderem Host laufen
    api_key: str | None = None
    tasks: SubAgentTasks = Field(default_factory=SubAgentTasks)


class EmbeddingConfig(BaseModel):
    model: str = "ollama/nomic-embed-text"
    api_base: str | None = None
    api_key: str | None = None
    enabled: bool = False


class StorageConfig(BaseModel):
    db_path: str = "./data/semanticswap.db"
    vector_store: str = "sqlite_naive"
    chunk_size: int = 1000


class CompressionConfig(BaseModel):
    keep_recent_messages: int = 4
    low_priority_visible: int = 3  # jüngere Low-Prio-Segmente bleiben sichtbar (ADR-011)
    # Iteration C: neuerer Fakt verdrängt älteren (gleiches Subjekt+Prädikat)
    temporal_supersede: bool = False
    # Iteration B: stehendes Nutzerprofil, immer im ARCHIVE-Prompt sichtbar
    user_profile: bool = False
    # Prio 2: Entitäts-Varianten kanonisieren (Graph-Dedup + Verdrängung)
    entity_resolution: bool = False


class RetrievalConfig(BaseModel):
    enabled: bool = True
    top_k: int = 3
    min_score: float = 0.35
    max_injection_tokens: int = 2000
    active_tool: bool = True  # retrieve_archived_memory-Tool (ADR-010)
    # Hybrid: FTS5-Stichwortsuche + Vektoren, verschmolzen per RRF.
    # min_score gilt dann nicht (RRF-Ränge statt Cosine-Scores).
    hybrid: bool = False
    # Graph-Expansion (Iteration D): Treffer ziehen Nachbar-Segmente mit,
    # die im Wissens-Graph Entitäten teilen (gegen multi-session-Fragen).
    graph_expansion: bool = False
    graph_expansion_limit: int = 3
    # Re-Ranker (Prio 1): LLM-Listwise-Neubewertung der Kandidaten. Holt
    # rerank_candidates Kandidaten, sortiert per LLM, behält top_k.
    rerank: bool = False
    rerank_candidates: int = 12
    rerank_model: str | None = None  # None = Standard je Backend
    # "cross_encoder" (ONNX, verlässlich) oder "llm" (Listwise, schwach bei
    # kleinen Modellen — nur für Rückwärtskompatibilität).
    rerank_backend: str = "cross_encoder"
    # Query-Decomposition (Prio 3): Multi-Hop-Frage in Teilfragen zerlegen,
    # jede suchen, Kandidaten per RRF verschmelzen (gegen multi-session).
    query_decompose: bool = False
    query_decompose_max: int = 3


class AppConfig(BaseModel):
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    main_llm: MainLLMConfig = Field(default_factory=MainLLMConfig)
    sub_agents: SubAgentsConfig = Field(default_factory=SubAgentsConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


def load_config(path: str | Path) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)
