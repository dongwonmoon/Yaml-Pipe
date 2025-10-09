"""
State management for the VectorFlow pipeline.

This module provides the StateManager class, which is responsible for tracking
the state of processed files. It works by storing a hash of each file that
has been processed, allowing the pipeline to skip files that have not changed
since the last run, making the process more efficient.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages the state of the pipeline, tracking processed files and their hashes.

    The state is persisted to a JSON file on disk, which stores a dictionary
    mapping file paths to their SHA256 hashes.
    """

    def __init__(self, state_file_path: str = ".vectorflow_state.json"):
        """
        Initializes the StateManager.

        Args:
            state_file_path (str): The path to the JSON file where the state
                                   is stored.
        """
        self.state_file_path = Path(state_file_path)
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """
        Loads the state from the JSON file if it exists, otherwise returns a new state.

        Returns:
            Dict: The loaded or newly created state dictionary.
        """
        if self.state_file_path.exists():
            logger.debug(f"Loading existing state from '{self.state_file_path}'")
            try:
                with open(self.state_file_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading state file: {e}. Starting with a fresh state.", exc_info=True)
                return {"processed_files": {}}

        logger.debug(f"No state file found. Creating new state.")
        return {"processed_files": {}}

    def save_state(self):
        """
        Saves the current state to the JSON file.
        """
        logger.debug(f"Saving state to '{self.state_file_path}'")
        try:
            with open(self.state_file_path, "w") as f:
                json.dump(self.state, f, indent=4)
            logger.info(f"Pipeline state saved successfully to '{self.state_file_path}'.")
        except IOError as e:
            logger.error(f"Error saving state file: {e}", exc_info=True)

    def get_file_hash(self, file_path: Path) -> str:
        """
        Computes the SHA256 hash of a file.

        Args:
            file_path (Path): The path to the file to hash.

        Returns:
            str: The hex digest of the file's hash.
        """
        hash_obj = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read the file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except (IOError, FileNotFoundError) as e:
            logger.error(f"Could not compute hash for file {file_path}: {e}", exc_info=True)
            return ""

    def has_changed(self, file_path_str: str) -> bool:
        """
        Checks if a file has changed since the last time it was processed.

        A file is considered changed if its current hash does not match the hash
        stored in the state, or if it is not in the state at all.

        Args:
            file_path_str (str): The string path of the file to check.

        Returns:
            bool: True if the file has changed or is new, False otherwise.
        """
        current_hash = self.get_file_hash(Path(file_path_str))
        if not current_hash:
            return False  # If hashing failed, treat as unchanged to be safe

        last_hash = self.state["processed_files"].get(file_path_str)

        changed = current_hash != last_hash
        if changed:
            logger.debug(f"Change detected for file '{file_path_str}' (new hash: {current_hash[:7]}...).")
        return changed

    def update_state(self, file_path_str: str):
        """
        Updates the state with the current hash of a file.

        This should be called after a file has been successfully processed.

        Args:
            file_path_str (str): The string path of the file to update.
        """
        current_hash = self.get_file_hash(Path(file_path_str))
        if current_hash:
            self.state["processed_files"][file_path_str] = current_hash
            logger.debug(f"Updated state for '{file_path_str}' with hash {current_hash[:7]}...")
