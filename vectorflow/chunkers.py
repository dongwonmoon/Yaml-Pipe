from abc import ABC, abstractmethod
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Chunks a single text into a list of text chunks."""
        pass


class RecursiveCharacterChunker(BaseChunker):
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        """Initializes the chunker with a specific chunk size and overlap."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """Splits the text into chunks."""
        logger.info(
            f"Splitting text with chunk size={self.chunk_size} and overlap={self.chunk_overlap}."
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        return text_splitter.split_text(text)
