import streamlit as st
import yaml
from pathlib import Path
import logging
import chromadb
import lancedb

from yamlpipe.core.pipeline import run_pipeline
from yamlpipe.core.factory import build_component, EMBEDDER_REGISTRY
from yamlpipe.utils.config import load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StreamlitLogHandler(logging.Handler):
    """Custom logging handler to display logs in a Streamlit container."""

    def __init__(self, container):
        super().__init__()
        self.container = container

    def emit(self, record):
        self.container.text(self.format(record))


class Searcher:
    """A class to perform searches on a vector database."""

    def __init__(self, embedder_config: dict, sink_config: dict):
        self.embedder = build_component(embedder_config, EMBEDDER_REGISTRY)
        self.sink_config = sink_config
        self.retriever = self._init_retriever()

    def _init_retriever(self):
        sink_type = self.sink_config["type"]
        if sink_type == "chromadb":
            client = chromadb.HttpClient(
                host=self.sink_config["config"]["host"],
                port=self.sink_config["config"]["port"],
            )
            return client.get_collection(
                name=self.sink_config["config"]["collection_name"]
            )
        elif sink_type == "lancedb":
            db = lancedb.connect(self.sink_config["config"]["uri"])
            return db.open_table(self.sink_config["config"]["table_name"])
        else:
            raise ValueError(f"Unsupported sink type: {sink_type}")

    def search(self, query: str, k: int = 3):
        query_vector = self.embedder.embed([query])[0]
        sink_type = self.sink_config["type"]
        if sink_type == "chromadb":
            return self.retriever.query(
                query_embeddings=[query_vector.tolist()], n_results=k
            )
        elif sink_type == "lancedb":
            return self.retriever.search(query_vector).limit(k).to_df()


def create_temp_pipeline_config(source_config: dict) -> str:
    """Creates a temporary pipeline configuration file."""
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
                "collection_name": "my_collection",
            },
        },
    }
    config_template["source"] = source_config

    temp_dir = Path("temp_ui")
    temp_dir.mkdir(exist_ok=True)
    config_path = temp_dir / "temp_pipeline.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_template, f)

    return str(config_path)


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="YamlPipe Dashboard", layout="wide")
    st.title("🚀 YamlPipe: AI Data Pipeline Dashboard")
    st.markdown(
        """Use this dashboard to run and test the core features of YamlPipe. 
    Select a data source, run the pipeline, and then ask questions to the generated vector database!"""
    )

    # --- 1. Data Source Selection ---
    st.header("1. Select Data Source")
    source_type = st.radio(
        "Select data source type:",
        ("Local File Upload", "Website URL"),
        horizontal=True,
    )

    source_config = None
    if source_type == "Local File Upload":
        uploaded_files = st.file_uploader(
            "Upload documents", accept_multiple_files=True
        )
        if uploaded_files:
            upload_dir = Path("temp_ui/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            for uploaded_file in uploaded_files:
                with open(upload_dir / uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            source_config = {
                "type": "local_files",
                "config": {"path": str(upload_dir), "glob_pattern": "*.*"},
            }

    elif source_type == "Website URL":
        url = st.text_input(
            "Enter website URL",
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
        )
        if url:
            source_config = {"type": "web", "config": {"url": url}}

    # --- 2. Run Pipeline ---
    st.header("2. Run Pipeline")
    if source_config:
        if st.button("▶️ Run Pipeline"):
            temp_config_path = create_temp_pipeline_config(source_config)
            st.info(f"Temporary config file created: {temp_config_path}")

            with st.spinner("Pipeline is running..."):
                log_container = st.expander(
                    "View Real-time Logs", expanded=True
                )
                streamlit_handler = StreamlitLogHandler(log_container)
                logging.getLogger().addHandler(streamlit_handler)

                try:
                    run_pipeline(config_path=temp_config_path)
                    st.success("🎉 Pipeline executed successfully!")
                    config = load_config(temp_config_path)
                    st.session_state["sink_config"] = config["sink"]
                    st.session_state["embedder_config"] = config["embedder"]
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    logging.getLogger().removeHandler(streamlit_handler)

    # --- 3. Search Test ---
    st.header("3. Search Test")
    if "sink_config" in st.session_state:
        st.info("Pipeline has been executed. You can now test the search.")
        query = st.text_input("Ask a question:")

        if query:
            try:
                with st.spinner("Searching..."):
                    searcher = Searcher(
                        st.session_state["embedder_config"],
                        st.session_state["sink_config"],
                    )
                    results = searcher.search(query)
                    st.subheader("🔍 Search Results (Top 3)")
                    if searcher.sink_config["type"] == "chromadb":
                        for i, (doc, meta) in enumerate(
                            zip(
                                results["documents"][0], results["metadatas"][0]
                            )
                        ):
                            st.markdown(
                                f"**{i+1}. Source: `{meta.get('source', 'N/A')}`**"
                            )
                            st.info(doc)
                    elif searcher.sink_config["type"] == "lancedb":
                        for index, row in results.iterrows():
                            st.markdown(
                                f"**{index+1}. Source: `{row.get('source', 'N/A')}`**"
                            )
                            st.info(row["text"])
            except Exception as e:
                st.error(f"An error occurred during search: {e}")
    else:
        st.warning("Please run a pipeline first.")


if __name__ == "__main__":
    main()
