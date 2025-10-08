from .sources import LocalFileSource
from .chunkers import RecursiveCharacterChunker
from .embedders import SentenceTransformerEmbedder
from .sinks import LanceDBSink

SOURCE_REGISTRY = {
    "local_files": LocalFileSource,
}

CHUNKER_REGISTRY = {
    "recursive_character": RecursiveCharacterChunker,
}

EMBEDDER_REGISTRY = {
    "sentence_transformer": SentenceTransformerEmbedder,
}

SINK_REGISTRY = {"lancedb": LanceDBSink}


def build_component(component_config: dict, registry: dict):
    """
    설정 딕셔너리와 레지스트리를 받아, 해당하는 컴포넌트(클래스 객체)를 생성합니다.

    Args:
        component_config: YAML 파일의 source, chunker, embedder 섹션 딕셔너리
                          (예: {'type': 'local_files', 'config': {'path': '...'}})
        registry: 사용할 레지스트리 (예: SOURCE_REGISTRY)
    """
    component_type = component_config.get("type", "")
    config = component_config.get("config", {})

    if not component_type:
        raise ValueError("설정에 'type'이 지정되지 않았습니다.")

    component_class = registry.get(component_type)
    if not component_class:
        raise ValueError(f"'{component_type}'은(는) 유효한 타입이 아닙니다.")

    return component_class(**config)
