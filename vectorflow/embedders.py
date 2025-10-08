from abc import ABC, abstractmethod
import numpy as np
import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, chunks: list[str]) -> np.ndarray:
        """Embeds a list of text chunks and returns a list of embeddings as a NumPy array."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Loads a SentenceTransformer model by its model name."""
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Embedder model loaded: {self.model_name}")

    def embed(self, chunks: list[str]) -> np.ndarray:
        """Converts text chunks into embeddings using the pre-loaded model."""
        return self.model.encode(chunks)
