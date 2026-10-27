"""Structured logging configuration."""
import logging
import sys
from src.core.config import get_settings


def setup_logging() -> logging.Logger:
    """Configures root logger with standard formatting."""
    settings = get_settings()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logger = logging.getLogger("debate_service")
    logger.setLevel(log_level)
    return logger

