from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Production RAG API"
    # Application
    app_env: str = "development"
    environment: str = "development"
    primary_model: str = "anthropic/claude-3-5-sonnet"
    debug: bool = False
    log_level: str = "INFO"
    rate_limit: str = "10/minute"
    rate_limit_default: str = "10/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3
    
    # LangSmith / LangChain Tracing
    langsmith_tracing: bool = True
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "Production RAG"
    
    # LLM & Database Keys
    openrouter_api_key: str = ""
    anthropic_api_key: str = ""
    database_url: str = ""
    api_key: str = "dev-key"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        return self.app_env == "production" or self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance – loaded once, reused everywhere."""
    return Settings()
