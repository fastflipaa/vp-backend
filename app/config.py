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
    OPENAI_API_KEY: str = ""
    LEARNING_INJECTION_ENABLED: bool = False  # Kill switch: set True to enable lesson injection into prompts

    # Canary mode (Phase 17)
    CANARY_TAG: str = "v3-canary"
    CANARY_ENABLED: bool = False  # Set to True to activate canary routing

    # Old-lead outreach (Phase 20)
    OUTREACH_BATCH_SIZE: int = 75          # Daily batch: 50-100 configurable, default 75
    OUTREACH_MAX_PER_QUARTER: int = 2      # Max attempts per rolling 90-day window
    OUTREACH_MIN_INACTIVE_DAYS: int = 30   # Minimum days inactive before eligible

    # Lead scoring (Phase 21)
    LEAD_SCORE_HOT_THRESHOLD: int = 80      # Score >= 80 = hot-lead
    LEAD_SCORE_WARM_THRESHOLD: int = 50     # Score 50-79 = warm-lead, below 50 = cold-lead
    LEAD_SCORE_ENABLED: bool = True          # Kill switch for scoring
    SUMMARY_INTERACTION_THRESHOLD: int = 10  # Min interactions for conversation summary
    SUMMARY_REFRESH_INTERVAL: int = 5        # New interactions before summary refresh

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
