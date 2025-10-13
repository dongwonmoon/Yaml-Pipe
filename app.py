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
st.title("🚀 YamlPipe: AI Data Pipeline Dashboard")

st.markdown(
    """
Use this dashboard to run and test the core features of YamlPipe without the terminal. 
Select a data source, run the pipeline, and then ask questions directly to the generated vector database!
"""
)

# --- 1. Data Source Selection Section ---
st.header("1. Select Data Source")
source_type = st.radio(
    "What kind of data would you like to process?",
    ("Local File Upload", "Website URL"),
    horizontal=True,
)


# 임시 YAML 파일을 생성하고 관리하기 위한 함수
def create_temp_pipeline_config(source_config):
    """Function to create a temporary pipeline configuration"""
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
if source_type == "Local File Upload":
    uploaded_files = st.file_uploader(
        "Upload your document files (.txt, .md, .pdf, etc.)",
        accept_multiple_files=True,
    )
    if uploaded_files:
        # Create a temporary directory to store uploaded files
        upload_dir = Path("temp_ui/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save files
        for uploaded_file in uploaded_files:
            with open(upload_dir / uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Create config for LocalFileSource
        source_config = {
            "type": "local_files",
            "config": {"path": str(upload_dir), "glob_pattern": "*.*"},
        }

elif source_type == "Website URL":
    url = st.text_input(
        "Enter the URL of the website to process", "https://en.wikipedia.org/wiki/Artificial_intelligence"
    )
    if url:
        # Create config for WebSource
        source_config = {"type": "web", "config": {"url": url}}

# --- 2. Run Pipeline Section ---
st.header("2. Run Pipeline")

if source_config:
    if st.button("▶️ Run Pipeline"):
        # 1. Create temporary config file
        temp_config_path = create_temp_pipeline_config(source_config)
        st.info(f"Temporary config file created: {temp_config_path}")

        # 2. Run pipeline and display logs
        with st.spinner("Pipeline is running... Please wait..."):
            log_container = st.expander("View Real-time Logs", expanded=True)
            with log_container:
                # Capture logs in a list instead of printing to console
                logs = []

                def log_message(message):
                    logs.append(message)
                    st.text(message)  # Display logs in real-time as text

                try:
                    # Since run_pipeline uses logging, add a Streamlit handler
                    class StreamlitLogHandler(logging.Handler):
                        def __init__(self, container):
                            super().__init__()
                            self.container = container

                        def emit(self, record):
                            self.container.text(self.format(record))

                    # Add handler to the root logger
                    streamlit_handler = StreamlitLogHandler(log_container)
                    logging.getLogger().addHandler(streamlit_handler)

                    run_pipeline(config_path=temp_config_path)
                    st.success("🎉 Pipeline executed successfully!")

                    # Remove handler after use to prevent duplicate logging
                    logging.getLogger().removeHandler(streamlit_handler)

                    # Save sink and embedder info to session for the search feature
                    st.session_state["sink_config"] = load_config(temp_config_path)[
                        "sink"
                    ]
                    st.session_state["embedder_config"] = load_config(temp_config_path)[
                        "embedder"
                    ]

                except Exception as e:
                    st.error(f"An error occurred during pipeline execution: {e}")

# --- 3. Search Test Section ---
st.header("3. Search Test")

if "sink_config" in st.session_state:
    st.info("Pipeline has been executed. You can now test the search below.")
    query = st.text_input("Ask a question to the vector database:")

    if query:
        try:
            # Reuse the evaluation logic for searching
            with st.spinner("Searching..."):
                embedder = build_component(
                    st.session_state["embedder_config"], EMBEDDER_REGISTRY
                )
                sink_config = st.session_state["sink_config"]

                # Create DB client (referencing Evaluator logic)
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

                # Perform search
                query_vector = embedder.embed([query])[0]

                results = None
                if sink_config["type"] == "chromadb":
                    results = retriever.query(
                        query_embeddings=[query_vector.tolist()], n_results=3
                    )
                    st.subheader("🔍 Search Results (Top 3)")
                    for i, (doc, meta) in enumerate(
                        zip(results["documents"][0], results["metadatas"][0])
                    ):
                        st.markdown(f"**{i+1}. Source: `{meta.get('source', 'N/A')}`**")
                        st.info(doc)

                elif sink_config["type"] == "lancedb":
                    results = retriever.search(query_vector).limit(3).to_df()
                    st.subheader("🔍 Search Results (Top 3)")
                    for index, row in results.iterrows():
                        st.markdown(
                            f"**{index+1}. Source: `{row.get('source', 'N/A')}`**"
                        )
                        st.info(row["text"])

        except Exception as e:
            st.error(f"An error occurred during search: {e}")

else:
    st.warning("Please run a pipeline first to create the database.")
