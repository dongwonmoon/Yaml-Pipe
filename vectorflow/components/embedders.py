"""
Embedding components for the VectorFlow pipeline.

This module contains classes responsible for converting text chunks
into numerical vector embeddings using various models.
"""

from abc import ABC, abstractmethod
import numpy as np
import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for all embedder components."""

    @abstractmethod
    def embed(self, chunks: list[str]) -> np.ndarray:
        """
        Embeds a list of text chunks into a NumPy array of vectors.

        Args:
            chunks (list[str]): A list of text strings to be embedded.

        Returns:
            np.ndarray: A 2D NumPy array where each row is the vector embedding
                        for the corresponding text chunk.
        """
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """An embedder that uses the sentence-transformers library."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initializes the SentenceTransformerEmbedder by loading a model.

        Args:
            model_name (str): The name of the sentence-transformer model to load
                              from the Hugging Face Hub.
        """
        self.model_name = model_name
        logger.debug(f"Loading SentenceTransformer model: '{self.model_name}'")
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"SentenceTransformer model '{self.model_name}' loaded successfully.")

    def embed(self, chunks: list[str]) -> np.ndarray:
        """Converts text chunks into embeddings using the pre-loaded model."""
        logger.info(f"Embedding {len(chunks)} chunks using '{self.model_name}'...")
        embeddings = self.model.encode(chunks)
        logger.debug(f"Finished embedding. Output shape: {embeddings.shape}")
        return embeddings
