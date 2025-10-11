"""
Configuration loading utility for YamlPipe.

This module provides a function to safely load and parse a YAML configuration file.
"""

import yaml
from pathlib import Path
import logging
from pydantic import ValidationError
import sys

from .config_models import PipelineConfig

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Loads and validates a YAML configuration file from the specified path.

    If the file is not found, unreadable, or fails validation, it logs a
    detailed error and terminates the program.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the validated configuration.
    """
    path = Path(config_path)
    if not path.is_file():
        logger.error(f"Configuration file not found or is not a file: '{path}'")
        sys.exit(1)

    logger.debug(f"Attempting to load and validate configuration from: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not config:
                logger.error(f"Configuration file is empty: '{path}'")
                sys.exit(1)

        # Pydantic Validation
        PipelineConfig.model_validate(config)

        logger.info(
            f"Successfully loaded and validated configuration from: '{path}'"
        )
        return config

    except (yaml.YAMLError, IOError) as e:
        logger.error(
            f"Error reading or parsing YAML file '{path}': {e}", exc_info=True
        )
        sys.exit(1)
    except ValidationError as e:
        # Pydantic provides detailed, user-friendly error messages.
        logger.error(f"Configuration validation failed:\n{e}")
        sys.exit(1)
