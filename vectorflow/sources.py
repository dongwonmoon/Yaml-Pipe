from abc import ABC, abstractmethod
import requests
from bs4 import BeautifulSoup


class BaseSource(ABC):
    @abstractmethod
    def load_data(self) -> str:
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


class WebSource(BaseSource):
    def __init__(self, url: str):
        self.url = url

    def load_data(self) -> str:
        """
        초기화된 URL에서 HTML을 가져와 텍스트만 추출하여 반환합니다.
        """
        print(f"'{self.url}' URL에서 데이터를 로드합니다...")

        try:
            response = requests.get(self.url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())
            return "\n".join(line for line in lines if line)

        except requests.exceptions.RequestException as e:
            print(f"웹사이트에 접속하는 중 에러가 발생했습니다: {e}")
            return ""
