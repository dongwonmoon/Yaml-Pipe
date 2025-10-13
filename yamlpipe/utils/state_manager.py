"""
State management for the YamlPipe pipeline.

This module provides the StateManager class, which is responsible for tracking
the state of processed items. It works by storing a hash or identifier of each item
that has been processed, allowing the pipeline to skip items that have not changed
since the last run, making the process more efficient.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages the state of the pipeline, tracking processed items and their hashes.
    """

    def __init__(self, state_file_path: str = ".yamlpipe_state.json"):
        self.state_file_path = Path(state_file_path)
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if self.state_file_path.exists():
            logger.debug(f"Loading state from '{self.state_file_path}'")
            try:
                with open(self.state_file_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(
                    f"Error loading state file: {e}. Starting fresh.",
                    exc_info=True,
                )
                return self._get_initial_state()
        return self._get_initial_state()

    def _get_initial_state(self) -> Dict:
        logger.debug("Creating new state.")
        return {"processed_items": {}, "last_run_timestamp": None}

    def save_state(self):
        logger.debug(f"Saving state to '{self.state_file_path}'")
        try:
            with open(self.state_file_path, "w") as f:
                json.dump(self.state, f, indent=4)
            logger.info(f"Pipeline state saved to '{self.state_file_path}'.")
        except IOError as e:
            logger.error(f"Error saving state file: {e}", exc_info=True)

    def get_file_hash(self, file_path: Path) -> Optional[str]:
        hash_obj = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except (IOError, FileNotFoundError) as e:
            logger.error(
                f"Could not compute hash for file {file_path}: {e}",
                exc_info=True,
            )
            return None

    def has_changed(self, item_id: str, new_hash: Optional[str] = None) -> bool:
        """
        Checks if an item has changed since the last time it was processed.

        Args:
            item_id (str): The unique identifier of the item to check.
            new_hash (Optional[str]): The new hash of the item. If not provided,
                                       it will be calculated from the item_id (if it's a file path).

        Returns:
            bool: True if the item has changed or is new, False otherwise.
        """
        if new_hash is None:
            current_hash = self.get_file_hash(Path(item_id))
            if not current_hash:
                return False  # Treat as unchanged if hashing fails
        else:
            current_hash = new_hash

        last_hash = self.state["processed_items"].get(item_id)
        changed = current_hash != last_hash
        if changed:
            logger.debug(f"Change detected for item '{item_id}'.")
        return changed

    def update_file_state(self, item_id: str, new_hash: Optional[str] = None):
        """
        Updates the state with the current hash of an item.

        Args:
            item_id (str): The unique identifier of the item to update.
            new_hash (Optional[str]): The new hash of the item. If not provided,
                                       it will be calculated from the item_id (if it's a file path).
        """
        if new_hash is None:
            current_hash = self.get_file_hash(Path(item_id))
        else:
            current_hash = new_hash

        if current_hash:
            self.state["processed_items"][item_id] = current_hash
            logger.debug(f"Updated state for item '{item_id}'.")

    def get_last_run_timestamp(self) -> Optional[str]:
        return self.state.get("last_run_timestamp")

    def update_run_timestamp(self):
        self.state["last_run_timestamp"] = datetime.now(
            timezone.utc
        ).isoformat()
