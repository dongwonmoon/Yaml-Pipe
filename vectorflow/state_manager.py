import json
import hashlib
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class StateManager:
    def __init__(self, state_file_path: str = ".vectorflow_state.json"):
        """
        state:
            "processed_files": {
                "./data/file1.txt: "asdbasd..."
            }
        """
        self.state_file_path = Path(state_file_path)
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """
        Load state from json file
        """
        if self.state_file_path.exists():
            logger.info(f"Loading state from {self.state_file_path}")
            with open(self.state_file_path, "r") as f:
                return json.load(f)

        logger.info(f"Create new state file at {self.state_file_path}")
        return {"processed_files": {}}

    def save_state(self):
        """
        Save current state to json file
        """
        with open(self.state_file_path, "w") as f:
            json.dump(self.state, f, indent=4)
            logger.info(f"Saved state to {self.state_file_path}")

    def get_file_hash(self, file_path: Path) -> str:
        """
        Get hash of a file
        """
        hash_obj = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def has_changed(self, file_path_str: str) -> bool:
        """
        Check if a file has changed since last processing
        """
        current_hash = self.get_file_hash(Path(file_path_str))
        last_hash = self.state["processed_files"].get(file_path_str)

        return current_hash != last_hash

    def update_state(self, file_path_str: str):
        """
        Update state with current hash of a file
        """
        current_hash = self.get_file_hash(Path(file_path_str))
        self.state["processed_files"][file_path_str] = current_hash
