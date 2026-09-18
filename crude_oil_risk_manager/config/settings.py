"""Application configuration loaded from environment variables and .env file via Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object for the crude oil risk manager."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # App
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8050
    APP_DEBUG: bool = False
    APP_TITLE: str = "Crude Oil Risk Manager"

    # Database
    DB_PATH: str = "db/crude_oil_risk.db"

    # Historical data storage
    HISTORICAL_DATA_DIR: str = "data/historical"

    # Live data adapter
    LIVE_API_BASE_URL: str = ""
    LIVE_API_TOKEN: str = ""
    LIVE_POLL_INTERVAL_SECONDS: float = 2.0
    LIVE_STALENESS_THRESHOLD_SECONDS: float = 10.0

    # Historical data adapter
    HISTORICAL_API_BASE_URL: str = ""
    HISTORICAL_API_TOKEN: str = ""
    HISTORICAL_API_CALLS_PER_MINUTE: int = 7

    # Alerts - in-app
    ALERT_PORTFOLIO_PNL_STOP: float = -50000.0
    ALERT_STRUCTURE_MAX_LOSS: float = -10000.0
    ALERT_STALENESS_THRESHOLD_SECONDS: float = 10.0

    # Alerts - Microsoft Teams
    TEAMS_WEBHOOK_URL: str = ""
    TEAMS_ALERTS_ENABLED: bool = False

    # Risk defaults
    DEFAULT_VAR_CONFIDENCE: float = 0.95
    DEFAULT_CORRELATION_WINDOW: int = 60
    DEFAULT_MARGIN_LIMIT: float = 1000000.0

    # Contract roll warning
    ROLL_WARNING_DAYS_BEFORE_EXPIRY: int = 5


settings = Settings()
