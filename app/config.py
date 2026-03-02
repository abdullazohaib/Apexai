from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    APP_NAME: str = "AI Response Comparator & Synthesizer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/data.db"

    # Cache TTL in seconds
    CACHE_TTL: int = 3600

    # Scoring weights
    WEIGHT_KEYWORD: float = 0.30
    WEIGHT_LENGTH: float = 0.15
    WEIGHT_READABILITY: float = 0.20
    WEIGHT_COSINE: float = 0.20
    WEIGHT_FACTUAL: float = 0.15

    # Ideal response length range (words)
    IDEAL_MIN_WORDS: int = 50
    IDEAL_MAX_WORDS: int = 400

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(BASE_DIR / "logs" / "app.log")

    class Config:
        env_file = ".env"


settings = Settings()