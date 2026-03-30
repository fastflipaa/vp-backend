"""Vive Polanco Backend - Configuration via pydantic-settings.

All required fields have NO defaults. The process crashes at import time
if any environment variable is missing, providing fail-fast behavior.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")

    # Required -- no defaults. Process crashes if missing.
    ANTHROPIC_API_KEY: str
    NEO4J_URI: str           # e.g., bolt://neo4j-sgw8p8loh8rkr0tthnjbifr4:7687
    NEO4J_PASSWORD: str
    REDIS_URL: str           # e.g., redis://:password@vp-redis:6379/0
    GHL_TOKEN: str
    GHL_LOCATION_ID: str

    # Optional
    SLACK_WEBHOOK_URL: str = ""

    # Canary mode (Phase 17)
    CANARY_TAG: str = "v3-canary"
    CANARY_ENABLED: bool = False  # Set to True to activate canary routing

    # Derived Redis URLs for different databases
    @property
    def redis_broker_url(self) -> str:
        """Celery broker: db0"""
        return self._redis_db_url(0)

    @property
    def redis_result_url(self) -> str:
        """Celery result backend: db1"""
        return self._redis_db_url(1)

    @property
    def redis_cache_url(self) -> str:
        """Application cache (rate limits, dedup, locks): db2"""
        return self._redis_db_url(2)

    def _redis_db_url(self, db: int) -> str:
        base = self.REDIS_URL.rstrip("/")
        # Strip any existing /N suffix
        if base.split("/")[-1].isdigit():
            base = "/".join(base.split("/")[:-1])
        return f"{base}/{db}"


# Singleton -- crash on import if vars missing
settings = Settings()
