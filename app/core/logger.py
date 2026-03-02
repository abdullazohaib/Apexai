import logging
import logging.handlers
from pathlib import Path
from app.config import settings


def setup_logging() -> logging.Logger:
    """Configure application-wide logging with file rotation and console output."""
    Path(settings.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("ai_comparator")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Rotating file handler (10 MB, keep 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        settings.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()