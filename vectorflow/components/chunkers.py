from abc import ABC, abstractmethod
import logging
from typing import List

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

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


class MarkdownChunker(BaseChunker):
    def __init__(self):
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self._splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

    def chunk(self, document: Document) -> List[Document]:
        logger.info("Process document by using markdown splitter.")

        text_chunks = self._splitter.split_text(document.content)

        return [
            Document(
                content=chunk.page_content,
                metadata={**document.metadata.copy(), **chunk.metadata},
            )
            for chunk in text_chunks
        ]


class AdaptiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        self._markdown_chunker = MarkdownChunker()
        self._recursive_chunker = RecursiveCharacterChunker(chunk_size, chunk_overlap)

    def _decide_strategy(self, document: Document) -> str:
        content = document.content
        if (
            content.count("\n#") >= 2
            or content.count("\n##") >= 2
            or content.count("\n###") >= 2
        ):
            return "markdown"
        else:
            return "recursive"

    def chunk(self, document: Document) -> list[Document]:
        strategy = self._decide_strategy(document)

        if strategy == "markdown":
            return self._markdown_chunker.chunk(document)
        elif strategy == "recursive":
            return self._recursive_chunker.chunk(document)
