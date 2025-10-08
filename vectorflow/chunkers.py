from abc import ABC, abstractmethod

from langchain.text_splitter import RecursiveCharacterTextSplitter


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """
        하나의 텍스트를 입력받아 여러 개의 텍스트 청크 리스트로 반환합니다.
        """
        pass


class RecursiveCharacterChunker(BaseChunker):
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        """
        분할할 청크의 크기와 겹치는 크기를 초기화합니다.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """
        텍스트를 청크로 분할합니다.
        """

        print(f"chunk size={self.chunk_size} 설정으로 텍스트를 분할합니다.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        return text_splitter.split_text(text)
