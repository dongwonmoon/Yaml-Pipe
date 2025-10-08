import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    지정된 경로의 YAML 파일을 읽어서 파이썬 딕셔너리로 반환합니다.
    """
    print(f"설정 파일 로드: {config_path}")

    path = Path(config_path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"설정 파일을 찾을 수 없습니다: {path}")
        return {}
    except yaml.YAMLError as e:
        print(f"설정 파일을 파싱할 수 없습니다: {path}")
        return {}
