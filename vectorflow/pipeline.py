import pandas as pd
import logging
from .config import load_config
from .factory import (
    build_component,
    SOURCE_REGISTRY,
    CHUNKER_REGISTRY,
    EMBEDDER_REGISTRY,
    SINK_REGISTRY,
)
from .state_manager import StateManager


logger = logging.getLogger(__name__)


def run_pipeline(config_path: str):
    """Runs the entire embedding pipeline based on a configuration file."""
    state_manager = StateManager()

    config = load_config(config_path)
    if not config:
        logger.warning("Configuration is empty. Aborting pipeline.")
        return

    config["source"]["config"]["state_manager"] = state_manager

    source = build_component(config["source"], SOURCE_REGISTRY)
    chunker = build_component(config["chunker"], CHUNKER_REGISTRY)
    embedder = build_component(config["embedder"], EMBEDDER_REGISTRY)
    sink = build_component(config["sink"], SINK_REGISTRY)

    documents_to_process = source.load_data()

    if not documents_to_process:
        logger.warning("No documents to process. All up to date!")
        return

    all_chunks = []
    all_embeddings = []
    processed_paths = []

    for doc in documents_to_process:
        chunks = chunker.chunk(doc["content"])
        embeddings = embedder.embed(chunks)

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)
        processed_paths.append(doc["path"])

    final_data = pd.DataFrame({"text": all_chunks, "vector": all_embeddings})
    sink.sink(final_data)

    for path in processed_paths:
        state_manager.update_state(path)
    state_manager.save_state()

    logger.info("VectorFlow pipeline completed successfully.")
