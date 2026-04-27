import logging
import logging.config
from pathlib import Path


def setup_logger(config_path="config.yaml"):
    """Sets up the logging configuration."""
    Path("logs").mkdir(exist_ok=True)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "logs/app.log",
                "formatter": "standard",
                "level": "INFO",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    }

    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    return logger


logger = setup_logger()
