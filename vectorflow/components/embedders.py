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
    """
    An embedder that uses the sentence-transformers library from Hugging Face.

    This class handles loading a pre-trained model and using it to encode
    a batch of text chunks into dense vector embeddings.
    """

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
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        """Loads the SentenceTransformer model and handles potential errors."""
        logger.debug(f"Loading SentenceTransformer model: '{self.model_name}'")
        try:
            model = SentenceTransformer(self.model_name)
            logger.info(f"SentenceTransformer model '{self.model_name}' loaded successfully.")
            return model
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer model '{self.model_name}'. "
                f"Please ensure the model name is correct and you have an internet connection.",
                exc_info=True
            )
            raise e

    def embed(self, chunks: list[str]) -> np.ndarray:
        """Converts text chunks into embeddings using the pre-loaded model."""
        if not chunks:
            logger.warning("Embedder received an empty list of chunks. Returning empty array.")
            return np.array([])

        logger.info(f"Embedding {len(chunks)} chunks using '{self.model_name}'...")
        try:
            embeddings = self.model.encode(chunks, show_progress_bar=False)
            logger.info(f"Finished embedding {len(chunks)} chunks.")
            logger.debug(f"Output embedding shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"An error occurred during the embedding process: {e}", exc_info=True)
            raise e