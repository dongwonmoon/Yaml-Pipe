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

logger = logging.getLogger(__name__)


def run_pipeline(config_path: str):
    """Runs the entire embedding pipeline based on a configuration file."""
    config = load_config(config_path)
    if not config:
        logger.warning("Configuration is empty. Aborting pipeline.")
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

    logger.info("VectorFlow pipeline completed successfully.")
