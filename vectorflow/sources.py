from abc import ABC, abstractmethod


class BaseSource(ABC):
    @abstractmethod
    def load_data(self):
        """
        데이터 소스로부터 데이터를 로드하고 텍스트 내용을 반환.
        """
        pass


class LocalFileSource(BaseSource):
    def __init__(self, path: str):
        self.path = path

    def load_data(self) -> str:
        print(f"{self.path} 파일에서 데이터를 로드합니다...")

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {self.path}")
            return ""
