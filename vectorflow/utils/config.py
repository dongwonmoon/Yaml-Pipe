"""
Configuration loading utility for VectorFlow.

This module provides a function to safely load and parse a YAML configuration file.
"""

import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Loads a YAML file from the specified path and returns it as a dictionary.

    This function handles file not found errors and YAML parsing errors gracefully
    by logging the error and returning an empty dictionary.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration, or an empty dictionary
              if an error occurred.
    """
    logger.debug(f"Attempting to load configuration from: {config_path}")

    path = Path(config_path)

    if not path.is_file():
        logger.error(f"Configuration file not found at path: {path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from: {path}")
            return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file '{path}': {e}", exc_info=True)
        return {}
    except IOError as e:
        logger.error(f"Error reading configuration file '{path}': {e}", exc_info=True)
        return {}
