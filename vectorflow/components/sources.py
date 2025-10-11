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

        This method should handle fetching data, parsing it, and wrapping it
        in Document objects. It should also interact with a StateManager to
        avoid reprocessing unchanged data.

        Returns:
            List[Document]: A list of Document objects, each representing a loaded item.
        """
        pass

    @abstractmethod
    def test_connection(self):
        """
        Tests the connection to the data source to ensure it is accessible.

        Raises:
            Exception: If the connection test fails.
        """
        pass


class LocalFileSource(BaseSource):
    """
    Loads documents from the local filesystem.

    This source scans a directory for files matching a glob pattern and uses the
    `unstructured` library to parse их content. It tracks file modifications
    using a StateManager to process only new or changed files.
    """

    def __init__(
        self, path: str, glob_pattern: str, state_manager: StateManager
    ):
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
        if not self.path.is_dir():
            logger.error(
                f"Source path '{self.path}' is not a valid directory. Aborting."
            )
            return []

        all_files = [
            str(f) for f in self.path.glob(self.glob_pattern) if f.is_file()
        ]
        logger.debug(
            f"Found {len(all_files)} total files matching glob pattern."
        )

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
                elements = partition(filename=file_path_str)
                content = "\n\n".join([str(el) for el in elements])

                if not content.strip():
                    logger.warning(
                        f"File '{file_path_str}' is empty or contains no text. Skipping."
                    )
                    continue

                doc = Document(
                    content=content, metadata={"source": file_path_str}
                )
                loaded_data.append(doc)
                logger.debug(
                    f"Successfully loaded and partitioned file: {file_path_str}"
                )

            except Exception as e:
                logger.error(
                    f"Error processing file '{file_path_str}': {e}",
                    exc_info=True,
                )

        logger.info(f"Successfully loaded {len(loaded_data)} documents.")
        return loaded_data

    def test_connection(self):
        """Tests if the source directory exists and is accessible."""
        logger.info(
            f"Testing connection for LocalFileSource at path: {self.path}"
        )
        if not self.path.exists():
            raise FileNotFoundError(
                f"Source path '{self.path}' does not exist."
            )
        if not self.path.is_dir():
            raise NotADirectoryError(
                f"Source path '{self.path}' is not a directory."
            )
        logger.info("Connection to LocalFileSource successful.")


class WebSource(BaseSource):
    """
    Loads a document from a web URL.

    This source fetches the content of a single web page, parses the HTML to
    extract clean text, and returns it as one Document.
    """

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
        Fetches content from the URL, parses it, and returns a single Document.
        """
        logger.info(f"Fetching content from URL: {self.url}")
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())
            clean_text = "\n".join(line for line in lines if line)

            if not clean_text.strip():
                logger.warning(f"No text content found at URL: {self.url}")
                return []

            doc = Document(content=clean_text, metadata={"source": self.url})
            logger.info(
                f"Successfully fetched and parsed content from: {self.url}"
            )
            return [doc]

        except requests.exceptions.RequestException as e:
            logger.error(
                f"Failed to fetch content from URL '{self.url}': {e}",
                exc_info=True,
            )
            return []

    def test_connection(self):
        """Tests if the web URL is reachable by sending a HEAD request."""
        logger.info(f"Testing connection for WebSource at URL: {self.url}")
        try:
            response = requests.head(self.url, timeout=5)
            response.raise_for_status()
            logger.info("Connection to WebSource successful.")
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Failed to connect to URL '{self.url}': {e}", exc_info=True
            )
            raise ConnectionError(f"Failed to connect to URL: {self.url}")
