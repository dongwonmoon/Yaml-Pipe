from abc import ABC, abstractmethod
import logging
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..core.data_models import Document


logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Document]:
        """Chunks a single text into a list of text chunks."""
        pass


class RecursiveCharacterChunker(BaseChunker):
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        """Initializes the chunker with a specific chunk size and overlap."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def chunk(self, document: Document) -> list[Document]:
        """Splits the text into chunks."""
        logger.info(
            f"Splitting text with chunk size={self.chunk_size} and overlap={self.chunk_overlap}."
        )
        text_chunks = self._text_splitter.split_text(document.content)

        chunked_documents = []
        for i, text_chunk in enumerate(text_chunks):
            new_metadata = document.metadata.copy()
            new_metadata["chunk_index"] = i + 1

            chunked_doc = Document(content=text_chunk, metadata=new_metadata)
            chunked_documents.append(chunked_doc)

        return chunked_documents
