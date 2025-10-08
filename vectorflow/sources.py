from abc import ABC, abstractmethod
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict

from .state_manager import StateManager

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    @abstractmethod
    def load_data(self) -> List[Dict[str, str]]:
        """Loads data from a source and returns its text content."""
        pass


class LocalFileSource(BaseSource):
    def __init__(self, path: str, glob_pattern: str, state_manager: StateManager):
        self.path = Path(path)
        self.glob_pattern = glob_pattern
        self.state_manager = state_manager

    def load_data(self) -> List[Dict[str, str]]:
        """Loads data from a local file and returns its text content."""
        logger.info(f"Loading data from file: {self.path}")
        all_files = [str(f) for f in self.path.glob(self.glob_pattern) if f.is_file()]

        new_or_changed_files = [
            f for f in all_files if self.state_manager.has_changed(f)
        ]

        if not new_or_changed_files:
            logger.info("No new or changed files found.")
            return []

        logger.info(f"Found {len(new_or_changed_files)} new or changed files.")

        loaded_data = []
        for file_path_str in new_or_changed_files:
            with open(file_path_str, "r") as f:
                file_content = f.read()
                loaded_data.append({"path": file_path_str, "content": file_content})

        return loaded_data


class WebSource(BaseSource):
    def __init__(self, url: str):
        self.url = url

    def load_data(self) -> List[Dict[str, str]]:
        """Fetches HTML from the initialized URL and returns the extracted text."""
        logger.info(f"Loading data from URL: {self.url}")
        try:
            response = requests.get(self.url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())
            clean_text = "\n".join(line for line in lines if line)

            return [{"path": self.url, "content": clean_text}]

        except requests.exceptions.RequestException as e:
            logger.error(f"Error accessing website: {self.url}", exc_info=True)
            return []
