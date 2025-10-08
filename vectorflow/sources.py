from abc import ABC, abstractmethod
import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    @abstractmethod
    def load_data(self) -> str:
        """Loads data from a source and returns its text content."""
        pass


class LocalFileSource(BaseSource):
    def __init__(self, path: str):
        self.path = path

    def load_data(self) -> str:
        """Loads data from a local file and returns its text content."""
        logger.info(f"Loading data from file: {self.path}")
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"File not found: {self.path}")
            return ""


class WebSource(BaseSource):
    def __init__(self, url: str):
        self.url = url

    def load_data(self) -> str:
        """Fetches HTML from the initialized URL and returns the extracted text."""
        logger.info(f"Loading data from URL: {self.url}")
        try:
            response = requests.get(self.url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())
            return "\n".join(line for line in lines if line)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error accessing website: {self.url}", exc_info=True)
            return ""
