"""
Embedding components for the YamlPipe pipeline.

This module contains classes responsible for converting text chunks
into numerical vector embeddings using various models.
"""

from abc import ABC, abstractmethod
import numpy as np
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

import os

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for all embedder components."""

    @abstractmethod
    def embed(self, chunks: list[str]) -> np.ndarray:
        """
        Embeds a list of text chunks into a NumPy array of vectors.
        """
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    An embedder that uses the sentence-transformers library from Hugging Face.
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        logger.debug(f"Loading SentenceTransformer model: '{self.model_name}'")
        try:
            model = SentenceTransformer(self.model_name)
            logger.info(
                f"SentenceTransformer model '{self.model_name}' loaded successfully."
            )
            return model
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer model '{self.model_name}'.",
                exc_info=True,
            )
            raise

    def embed(self, chunks: list[str]) -> np.ndarray:
        if not chunks:
            logger.warning("Embedder received an empty list of chunks.")
            return np.array([])

        logger.info(
            f"Embedding {len(chunks)} chunks using '{self.model_name}'..."
        )
        try:
            embeddings = self.model.encode(chunks, show_progress_bar=False)
            logger.info(f"Finished embedding {len(chunks)} chunks.")
            return embeddings
        except Exception as e:
            logger.error(
                f"An error occurred during embedding: {e}", exc_info=True
            )
            raise


class OpenAIEmbedder(BaseEmbedder):
    """
    An embedder that uses the OpenAI API to generate embeddings.
    """

    def __init__(
        self, model_name: str = "text-embedding-3-small", api_key: str = None
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=self.api_key)
        logger.info(
            f"Initialized OpenAIEmbedder with model '{self.model_name}'."
        )

    def embed(self, chunks: list[str]) -> np.ndarray:
        if not chunks:
            logger.warning("Got an empty list of chunks.")
            return np.array([])

        logger.info(
            f"Embedding {len(chunks)} chunks using '{self.model_name}'..."
        )
        try:
            response = self.client.embeddings.create(
                input=chunks, model=self.model_name
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error while embedding: {e}", exc_info=True)
            raise
