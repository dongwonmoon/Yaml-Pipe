import pandas as pd
from .config import load_config
from .factory import (
    build_component,
    SOURCE_REGISTRY,
    CHUNKER_REGISTRY,
    EMBEDDER_REGISTRY,
    SINK_REGISTRY,
)


def run_pipeline(config_path: str = "pipeline.yml"):
    """
    설정 파일을 기반으로 전체 임베딩 파이프라인을 실행합니다.
    """
    config = load_config(config_path)
    if not config:
        print("🚨 설정 파일이 비어있어 파이프라인을 중단합니다.")
        return

    source = build_component(config["source"], SOURCE_REGISTRY)
    chunker = build_component(config["chunker"], CHUNKER_REGISTRY)
    embedder = build_component(config["embedder"], EMBEDDER_REGISTRY)
    sink = build_component(config["sink"], SINK_REGISTRY)

    text_content = source.load_data()
    chunks = chunker.chunk(text_content)
    embeddings = embedder.embed(chunks)

    final_data = pd.DataFrame({"text": chunks, "vector": list(embeddings)})
    sink.sink(final_data)

    print("🎉 VectorFlow 파이프라인이 성공적으로 완료되었습니다!")
