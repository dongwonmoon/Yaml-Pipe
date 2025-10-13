"""
Text chunking components for the YamlPipe pipeline.

This module provides different strategies for splitting large documents into
smaller, manageable chunks. This is a crucial step before generating embeddings.
"""

from abc import ABC, abstractmethod
import logging
from typing import List

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from ..utils.data_models import Document

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for all chunker components."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Document]:
        """
        Chunks a single document into a list of smaller documents.

        Args:
            document (Document): The document to be chunked.

        Returns:
            list[Document]: A list of chunked documents.
        """
        pass


class RecursiveCharacterChunker(BaseChunker):
    """
    A chunker that splits text recursively by a list of specified characters.

    This method is effective for maintaining semantic coherence by trying to
    split on sentence- and paragraph-level boundaries first.
    """

    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        """
        Initializes the chunker with a specific chunk size and overlap.

        Args:
            chunk_size (int): The maximum size of each chunk (measured by length).
            chunk_overlap (int): The number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        logger.debug(
            f"Initialized RecursiveCharacterChunker with size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk(self, document: Document) -> list[Document]:
        """Splits a document's content into chunks using a recursive character splitter."""
        source = document.metadata.get("source", "unknown")
        if not document.content or not document.content.strip():
            logger.warning(
                f"Document from source '{source}' is empty. Skipping chunking."
            )
            return []

        logger.debug(f"Recursively splitting document from source: {source}")
        text_chunks = self._text_splitter.split_text(document.content)

        chunked_documents = []
        for i, text_chunk in enumerate(text_chunks):
            new_metadata = document.metadata.copy()
            new_metadata["chunk_index"] = i + 1
            chunked_doc = Document(content=text_chunk, metadata=new_metadata)
            chunked_documents.append(chunked_doc)

        logger.debug(f"Created {len(chunked_documents)} chunks from source: {source}")
        return chunked_documents


class MarkdownChunker(BaseChunker):
    """
    A chunker that splits text based on Markdown headers.

    This is useful for preserving the structure of documentation or other
    Markdown-formatted text, using headers as logical section breaks.
    """

    def __init__(self):
        """Initializes the Markdown chunker."""
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self._splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
        logger.debug("Initialized MarkdownChunker")

    def chunk(self, document: Document) -> List[Document]:
        """Splits a document based on Markdown header structure."""
        source = document.metadata.get("source", "unknown")
        if not document.content or not document.content.strip():
            logger.warning(
                f"Document from source '{source}' is empty. Skipping chunking."
            )
            return []

        logger.debug(f"Splitting Markdown document from source: {source}")
        try:
            text_chunks = self._splitter.split_text(document.content)
            chunked_documents = [
                Document(
                    content=chunk.page_content,
                    metadata={**document.metadata.copy(), **chunk.metadata},
                )
                for chunk in text_chunks
            ]
            logger.debug(
                f"Created {len(chunked_documents)} chunks from source: {source}"
            )
            return chunked_documents
        except Exception as e:
            logger.error(
                f"Failed to split Markdown for source '{source}': {e}",
                exc_info=True,
            )
            return []


class AdaptiveChunker(BaseChunker):
    """
    A chunker that dynamically chooses a chunking strategy.

    It decides whether to use the MarkdownChunker or the RecursiveCharacterChunker
    based on the presence of Markdown headers in the document content.
    """

    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        """
        Initializes the AdaptiveChunker with underlying chunker instances.

        Args:
            chunk_size (int): The chunk size to be used by the recursive chunker.
            chunk_overlap (int): The chunk overlap for the recursive chunker.
        """
        self._markdown_chunker = MarkdownChunker()
        self._recursive_chunker = RecursiveCharacterChunker(chunk_size, chunk_overlap)
        logger.debug("Initialized AdaptiveChunker")

    def _decide_strategy(self, document: Document) -> str:
        """Heuristically determines the best chunking strategy for a document."""
        content = document.content
        if (
            content.count("\n# ") >= 2
            or content.count("\n## ") >= 2
            or content.count("\n### ") >= 2
        ):
            return "markdown"
        else:
            return "recursive"

    def chunk(self, document: Document) -> list[Document]:
        """Chunks the document using the adaptively chosen strategy."""
        strategy = self._decide_strategy(document)
        source = document.metadata.get("source", "unknown")

        logger.info(f"Using '{strategy}' chunking strategy for source: {source}")

        if strategy == "markdown":
            return self._markdown_chunker.chunk(document)
        else:  # strategy == "recursive"
            return self._recursive_chunker.chunk(document)
