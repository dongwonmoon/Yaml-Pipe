import streamlit as st
import yaml
from pathlib import Path
import os
import time
import logging
import chromadb

from yamlpipe.core.pipeline import run_pipeline
from yamlpipe.core.factory import build_component, EMBEDDER_REGISTRY, SINK_REGISTRY
from yamlpipe.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- UI 구성 ---

st.set_page_config(page_title="YamlPipe Dashboard", layout="wide")
st.title("🚀 YamlPipe: AI 데이터 파이프라인 대시보드")

st.markdown(
    """
이 대시보드를 사용하여 터미널 없이 YamlPipe의 핵심 기능을 실행하고 테스트해볼 수 있습니다. 
데이터 소스를 선택하고, 파이프라인을 실행한 뒤, 생성된 벡터 DB에 직접 질문해보세요!
"""
)

# --- 1. 데이터 소스 선택 섹션 ---
st.header("1. 데이터 소스 선택")
source_type = st.radio(
    "어떤 종류의 데이터를 처리할까요?",
    ("로컬 파일 업로드", "웹사이트 URL"),
    horizontal=True,
)


# 임시 YAML 파일을 생성하고 관리하기 위한 함수
def create_temp_pipeline_config(source_config):
    """임시 파이프라인 설정을 생성하는 함수"""
    # 기본 템플릿
    config_template = {
        "chunker": {
            "type": "adaptive",
            "config": {"chunk_size": 500, "chunk_overlap": 50},
        },
        "embedder": {
            "type": "sentence_transformer",
            "config": {"model_name": "jhgan/ko-sbert-nli"},
        },
        "sink": {
            "type": "chromadb",
            "config": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "my_server_collection",
            },
        },
    }
    # 소스 설정을 템플릿에 추가
    config_template["source"] = source_config

    # 임시 폴더 및 파일 경로 설정
    temp_dir = Path("temp_ui")
    temp_dir.mkdir(exist_ok=True)
    config_path = temp_dir / "temp_pipeline.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_template, f)

    return str(config_path)


source_config = None
if source_type == "로컬 파일 업로드":
    uploaded_files = st.file_uploader(
        "처리할 문서 파일들을 업로드하세요 (.txt, .md, .pdf 등)",
        accept_multiple_files=True,
    )
    if uploaded_files:
        # 업로드된 파일을 저장할 임시 폴더 생성
        upload_dir = Path("temp_ui/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # 파일 저장
        for uploaded_file in uploaded_files:
            with open(upload_dir / uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # LocalFileSource를 위한 설정 생성
        source_config = {
            "type": "local_files",
            "config": {"path": str(upload_dir), "glob_pattern": "*.*"},
        }

elif source_type == "웹사이트 URL":
    url = st.text_input(
        "처리할 웹사이트의 URL을 입력하세요", "https://ko.wikipedia.org/wiki/인공지능"
    )
    if url:
        # WebSource를 위한 설정 생성
        source_config = {"type": "web", "config": {"url": url}}

# --- 2. 파이프라인 실행 섹션 ---
st.header("2. 파이프라인 실행")

if source_config:
    if st.button("▶️ 파이프라인 실행하기"):
        # 1. 임시 설정 파일 생성
        temp_config_path = create_temp_pipeline_config(source_config)
        st.info(f"임시 설정 파일 생성: {temp_config_path}")

        # 2. 파이프라인 실행 및 로그 출력
        with st.spinner("파이프라인이 실행 중입니다... 잠시만 기다려주세요..."):
            log_container = st.expander("실시간 로그 보기", expanded=True)
            with log_container:
                # 간단한 로그 캡처를 위해 print 대신 리스트에 저장
                logs = []

                def log_message(message):
                    logs.append(message)
                    st.text(message)  # 실시간으로 로그를 화면에 텍스트로 출력

                try:
                    # run_pipeline이 로깅을 사용하므로, Streamlit 핸들러를 추가합니다.
                    class StreamlitLogHandler(logging.Handler):
                        def __init__(self, container):
                            super().__init__()
                            self.container = container

                        def emit(self, record):
                            self.container.text(self.format(record))

                    # 기존 로거에 핸들러 추가
                    streamlit_handler = StreamlitLogHandler(log_container)
                    logging.getLogger().addHandler(streamlit_handler)

                    run_pipeline(config_path=temp_config_path)
                    st.success("🎉 파이프라인 실행이 성공적으로 완료되었습니다!")

                    # 사용 후 핸들러 제거 (중복 로깅 방지)
                    logging.getLogger().removeHandler(streamlit_handler)

                    # 검색 기능을 위해 sink 정보를 세션에 저장
                    st.session_state["sink_config"] = load_config(temp_config_path)[
                        "sink"
                    ]
                    st.session_state["embedder_config"] = load_config(temp_config_path)[
                        "embedder"
                    ]

                except Exception as e:
                    st.error(f"파이프라인 실행 중 에러 발생: {e}")

# --- 3. 검색 테스트 섹션 ---
st.header("3. 검색 테스트")

if "sink_config" in st.session_state:
    st.info("파이프라인이 성공적으로 실행되어, 아래에서 검색을 테스트해볼 수 있습니다.")
    query = st.text_input("벡터 데이터베이스에 질문해보세요:")

    if query:
        try:
            # 평가(Evaluation) 로직 재활용
            with st.spinner("검색 중..."):
                embedder = build_component(
                    st.session_state["embedder_config"], EMBEDDER_REGISTRY
                )
                sink_config = st.session_state["sink_config"]

                # DB 클라이언트 생성 (Evaluator 로직 참고)
                retriever = None
                if sink_config["type"] == "chromadb":
                    client = chromadb.HttpClient(
                        host=sink_config["config"]["host"],
                        port=sink_config["config"]["port"],
                    )
                    retriever = client.get_collection(
                        name=sink_config["config"]["collection_name"]
                    )
                elif sink_config["type"] == "lancedb":
                    import lancedb

                    db = lancedb.connect(sink_config["config"]["uri"])
                    retriever = db.open_table(sink_config["config"]["table_name"])

                # 검색 수행
                query_vector = embedder.embed([query])[0]

                results = None
                if sink_config["type"] == "chromadb":
                    results = retriever.query(
                        query_embeddings=[query_vector.tolist()], n_results=3
                    )
                    st.subheader("🔍 검색 결과 (Top 3)")
                    for i, (doc, meta) in enumerate(
                        zip(results["documents"][0], results["metadatas"][0])
                    ):
                        st.markdown(f"**{i+1}. 출처: `{meta.get('source', 'N/A')}`**")
                        st.info(doc)

                elif sink_config["type"] == "lancedb":
                    results = retriever.search(query_vector).limit(3).to_df()
                    st.subheader("🔍 검색 결과 (Top 3)")
                    for index, row in results.iterrows():
                        st.markdown(
                            f"**{index+1}. 출처: `{row.get('source', 'N/A')}`**"
                        )
                        st.info(row["text"])

        except Exception as e:
            st.error(f"검색 중 에러 발생: {e}")

else:
    st.warning("먼저 파이프라인을 실행하여 데이터베이스를 생성해주세요.")
