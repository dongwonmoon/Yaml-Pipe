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
from abc import ABC, abstractmethod
import redis

logger = logging.getLogger(__name__)


class BaseStateManager(ABC):
    """
    An abstract base class for all state manager components.
    """

    @abstractmethod
    def load_state(self) -> Dict:
        """Load state from storage."""
        pass

    @abstractmethod
    def save_state(self, state: Dict):
        """Saves the given state to storage."""
        pass


class JSONStateManager(BaseStateManager):
    """
    A state manager that stores state in a JSON file.
    """

    def __init__(self, path: str = ".yamlpipe_state.json"):
        """Initialize JSON path."""
        self.state_file_path = Path(path)

    def load_state(self) -> Dict:
        """
        If the JSON file exists, it is read, otherwise a new state is created.
        """
        if not self.state_file_path.exists():
            logger.debug("Creating new state as state file does not exist.")
            return {"processed_items": {}, "last_run_timestamp": None}

        logger.debug(f"Loading state from '{self.state_file_path}'")
        try:
            with open(self.state_file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.error("Error loading state file. Starting fresh.", exc_info=True)
            return {"processed_items": {}, "last_run_timestamp": None}

    def save_state(self, state: Dict):
        """Save the given state to the JSON file."""
        logger.debug(f"Saving state to '{self.state_file_path}'")
        try:
            with open(self.state_file_path, "w") as f:
                json.dump(state, f, indent=4)
            logger.info(f"Pipeline state saved to '{self.state_file_path}'.")
        except IOError as e:
            logger.error(f"Error saving state file: {e}", exc_info=True)


class RedisStateManager(BaseStateManager):
    """
    A class that manages the state of the pipeline using Redis.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        state_key: str = "yamlpipe_state",
    ):
        """
        Initialize the Redis state manager.
        """
        try:
            self.redis_client = redis.Redis(
                host=host, port=port, db=db, decode_responses=True
            )
            self.state_key = state_key
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Error connecting to Redis: {e}", exc_info=True)
            raise

    def load_state(self) -> Dict:
        """
        Load the state from Redis.
        """
        logger.debug(f"Loading state from Redis key '{self.state_key}'")
        try:
            existing_state = self.redis_client.get(self.state_key)
            if existing_state:
                return json.loads(existing_state)
            else:
                return {"processed_items": {}, "last_run_timestamp": None}
        except redis.exceptions.RedisError as e:
            logger.error(f"Error loading state from Redis: {e}", exc_info=True)
            return {"processed_items": {}, "last_run_timestamp": None}

    def save_state(self, state: Dict):
        """
        Save the state to Redis.
        """
        logger.debug(f"Saving state to Redis key '{self.state_key}'")
        try:
            self.redis_client.set(self.state_key, json.dumps(state))
            logger.info(f"Pipeline state saved to Redis key '{self.state_key}'.")
        except redis.exceptions.RedisError as e:
            logger.error(f"Error saving state to Redis: {e}", exc_info=True)


class StateManager:
    """
    Inject a backend (such as JSONStateManager) that will handle the actual state saving/loading.
    """

    def __init__(self, backend: BaseStateManager):
        self.backend = backend
        self.state = self.backend.load_state()

    def save(self):
        """Save current state through backend"""
        self.backend.save_state(self.state)

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
        self.state["last_run_timestamp"] = datetime.now(timezone.utc).isoformat()
