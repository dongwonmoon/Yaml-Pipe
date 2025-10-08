from abc import ABC, abstractmethod
import numpy as np

from sentence_transformers import SentenceTransformer


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, chunks: list[str]) -> np.ndarray:
        """
        텍스트 청크 리스트를 입력받아 NumPy 배열 형태의 임베딩 리스트를 반환합니다.
        """
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        사용할 모델의 이름을 받아 SentenceTransformer 모델을 로드합니다.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        print(f"Embedder 모델 '{self.model_name} 로드 완료.")

    def embed(self, chunks: list[str]) -> np.ndarray:
        """
        미리 로드된 모델을 사용하여 텍스트 청크들을 임베딩으로 변환합니다.
        """
        return self.model.encode(chunks)
