from abc import ABC, abstractmethod
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict
from unstructured.partition.auto import partition

from ..core.state_manager import StateManager
from ..core.data_models import Document

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    @abstractmethod
    def load_data(self) -> List[Document]:
        """Loads data from a source and returns its text content."""
        pass


class LocalFileSource(BaseSource):
    def __init__(self, path: str, glob_pattern: str, state_manager: StateManager):
        self.path = Path(path)
        self.glob_pattern = glob_pattern
        self.state_manager = state_manager

    def load_data(self) -> List[Document]:
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
            try:
                elements = partition(filename=file_path_str)
                content = "\n\n".join([str(el) for el in elements])

                doc = Document(content=content, metadata={"source": file_path_str})
                loaded_data.append(doc)
                logger.info(f"Loaded file: {file_path_str}")

            except Exception as e:
                logger.error(f"Error loading file: {file_path_str}", exc_info=True)

        return loaded_data


class WebSource(BaseSource):
    def __init__(self, url: str):
        self.url = url

    def load_data(self) -> List[Document]:
        """Fetches HTML from the initialized URL and returns the extracted text."""
        logger.info(f"Loading data from URL: {self.url}")
        try:
            response = requests.get(self.url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())
            clean_text = "\n".join(line for line in lines if line)

            doc = Document(content=clean_text, metadata={"source": self.url})
            return [doc]

        except requests.exceptions.RequestException as e:
            logger.error(f"Error accessing website: {self.url}", exc_info=True)
            return []
