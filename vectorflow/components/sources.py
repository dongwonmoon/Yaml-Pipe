"""
Data source components for the VectorFlow pipeline.

This module contains classes for loading data from various sources,
such as local files and web pages. Each source is responsible for
fetching raw data and converting it into a list of Document objects.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import logging
from typing import List
from unstructured.partition.auto import partition

from ..core.state_manager import StateManager
from ..core.data_models import Document

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    """Abstract base class for all data source components."""

    @abstractmethod
    def load_data(self) -> List[Document]:
        """
        Loads data from the configured source and returns a list of Documents.

        Returns:
            List[Document]: A list of Document objects, each representing a loaded item.
        """
        pass

    @abstractmethod
    def test_connection(self):
        """Tests the connection to the data source."""
        pass


class LocalFileSource(BaseSource):
    """Loads documents from the local filesystem."""

    def __init__(self, path: str, glob_pattern: str, state_manager: StateManager):
        """
        Initializes the LocalFileSource.

        Args:
            path (str): The directory path to search for files.
            glob_pattern (str): The glob pattern to match files within the directory.
            state_manager (StateManager): The state manager to track file changes.
        """
        self.path = Path(path)
        self.glob_pattern = glob_pattern
        self.state_manager = state_manager
        logger.debug(
            f"Initialized LocalFileSource with path='{self.path}' and glob='{self.glob_pattern}'"
        )

    def load_data(self) -> List[Document]:
        """
        Loads all files matching the glob pattern from the specified path,
        filters out files that have not changed since the last run, and
        returns a list of Document objects for the new or modified files.
        """
        logger.info(
            f"Scanning for files in '{self.path}' with pattern '{self.glob_pattern}'."
        )
        all_files = [str(f) for f in self.path.glob(self.glob_pattern) if f.is_file()]
        logger.debug(f"Found {len(all_files)} total files matching glob pattern.")

        # Filter files based on whether they have changed
        new_or_changed_files = [
            f for f in all_files if self.state_manager.has_changed(f)
        ]

        if not new_or_changed_files:
            logger.info("No new or changed files detected.")
            return []

        logger.info(
            f"Found {len(new_or_changed_files)} new or changed files to process."
        )

        loaded_data = []
        for file_path_str in new_or_changed_files:
            try:
                logger.debug(f"Partitioning file: {file_path_str}")
                # Use unstructured.io to partition the file into elements
                elements = partition(filename=file_path_str)
                content = "\n\n".join([str(el) for el in elements])

                doc = Document(content=content, metadata={"source": file_path_str})
                loaded_data.append(doc)
                logger.info(
                    f"Successfully loaded and partitioned file: {file_path_str}"
                )

            except Exception as e:
                logger.error(f"Error processing file: {file_path_str}", exc_info=True)

        return loaded_data

    def test_connection(self):
        logger.info(f"Testing connection for LocalFileSource at path: {self.path}")
        if not self.path.exists():
            raise ValueError(f"Path '{self.path}' does not exist.")
        if not self.path.is_dir():
            raise ValueError(f"'{self.path}' is not a directory.")

        logger.info("Connection to LocalFileSource successful.")


class WebSource(BaseSource):
    """Loads documents from a web URL."""

    def __init__(self, url: str):
        """
        Initializes the WebSource.

        Args:
            url (str): The URL of the web page to load.
        """
        self.url = url
        logger.debug(f"Initialized WebSource with URL: {self.url}")

    def load_data(self) -> List[Document]:
        """
        Fetches the content from the specified URL, parses the text,
        and returns it as a single Document.
        """
        logger.info(f"Fetching content from URL: {self.url}")
        try:
            response = requests.get(self.url)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Use BeautifulSoup to parse HTML and extract clean text
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            clean_text = "\n".join(line for line in lines if line)

            doc = Document(content=clean_text, metadata={"source": self.url})
            logger.info(f"Successfully fetched and parsed content from: {self.url}")
            return [doc]

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch content from URL: {self.url}", exc_info=True)
            return []

    def test_connection(self):
        logger.info(f"Testing connection for WebSource at URL: {self.url}")
        try:
            response = requests.head(self.url, timeout=5)
            response.raise_for_status()
            logger.info("Connection to WebSource successful.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to URL: {self.url}", exc_info=True)
            raise ConnectionError(f"Failed to connect to URL: {self.url} - {e}")
