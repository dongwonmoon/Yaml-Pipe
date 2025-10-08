import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Loads a YAML file from the specified path and returns it as a dictionary."""
    logger.info(f"Loading configuration from: {config_path}")

    path = Path(config_path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {path}", exc_info=True)
        return {}
