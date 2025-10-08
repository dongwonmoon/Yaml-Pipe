import lancedb
from pydantic import BaseModel
from lancedb.pydantic import pydantic_to_schema, Vector

# 이제 부품을 직접 임포트할 필요가 없습니다. 공장이 다 알아서 해주니까요!
from .config import load_config
from .factory import (
    build_component,
    SOURCE_REGISTRY,
    CHUNKER_REGISTRY,
    EMBEDDER_REGISTRY,
)


def run_pipeline(config_path: str = "pipeline.yaml"):
    """
    설정 파일을 기반으로 전체 임베딩 파이프라인을 실행합니다.
    """
    print("🚀 VectorFlow 파이프라인을 시작합니다...")

    # 1. 레시피 카드(YAML)를 읽는다.
    config = load_config(config_path)
    if not config:
        print("🚨 설정 파일이 비어있거나 올바르지 않아 파이프라인을 중단합니다.")
        return

    # 2. 공장에 레시피를 전달하여 필요한 부품들을 생산한다.
    print("🏭 부품 공장을 가동합니다...")
    source = build_component(config["source"], SOURCE_REGISTRY)
    chunker = build_component(config["chunker"], CHUNKER_REGISTRY)
    embedder = build_component(config["embedder"], EMBEDDER_REGISTRY)

    # 3. 생산된 부품들로 작업을 수행한다.
    print("✨ 파이프라인 작업을 시작합니다...")
    text_content = source.load_data()
    chunks = chunker.chunk(text_content)
    embeddings = embedder.embed(chunks)

    # 4. 최종 산출물을 저장한다. (이 부분은 나중에 'Sink' 부품으로 만들 수 있겠죠?)
    db_uri = "./lancedb_poc"
    db = lancedb.connect(db_uri)
    vector_dimensions = embeddings.shape[1]

    class Document(BaseModel):
        text: str
        vector: Vector(vector_dimensions)

    pyarrow_schema = pydantic_to_schema(Document)
    table_name = "vectorflow_docs"
    db.drop_table(table_name, ignore_missing=True)
    table = db.create_table(table_name, schema=pyarrow_schema)
    data_to_add = [
        {"text": chunk, "vector": embeddings[i]} for i, chunk in enumerate(chunks)
    ]
    table.add(data_to_add)
    print("✅ 벡터 DB 저장 완료")

    print("🎉 VectorFlow 파이프라인 실행이 성공적으로 완료되었습니다!")
