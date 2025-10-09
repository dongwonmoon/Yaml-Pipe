"""
Core pipeline orchestration module.

This module defines the main function `run_pipeline` that reads a YAML configuration,
builds the necessary components (source, chunker, embedder, sink), and executes
the ETL pipeline to process text data into vector embeddings.
"""

import logging
from ..utils.config import load_config
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
    """
    Runs the entire embedding pipeline based on a configuration file.

    This function orchestrates the following steps:
    1. Load the state manager to track processed files.
    2. Load the YAML configuration file.
    3. Build the source, chunker, embedder, and sink components using the factory.
    4. Load data from the source, filtering out unchanged files.
    5. Chunk the loaded documents into smaller pieces.
    6. Generate vector embeddings for each chunk.
    7. Sink the final chunks with their embeddings into the target data store.
    8. Update and save the state for the processed files.

    Args:
        config_path (str): The path to the pipeline's YAML configuration file.
    """
    logger.info(f"VectorFlow pipeline starting with config: {config_path}")

    # 1. Initialize State Manager
    state_manager = StateManager()

    # 2. Load Configuration
    config = load_config(config_path)
    if not config:
        logger.error("Configuration is empty or could not be loaded. Aborting pipeline.")
        return

    # Inject the state manager into the source component's configuration
    config["source"]["config"]["state_manager"] = state_manager

    # 3. Build Components
    logger.info("Building pipeline components...")
    try:
        source = build_component(config["source"], SOURCE_REGISTRY)
        chunker = build_component(config["chunker"], CHUNKER_REGISTRY)
        embedder = build_component(config["embedder"], EMBEDDER_REGISTRY)
        sink = build_component(config["sink"], SINK_REGISTRY)
        logger.info("All components built successfully.")
    except ValueError as e:
        logger.error(f"Error building components: {e}", exc_info=True)
        return

    # 4. Load Data
    logger.info(f"Loading data from source: {source.__class__.__name__}")
    documents_to_process = source.load_data()

    if not documents_to_process:
        logger.info("No new or modified documents to process. Pipeline finished.")
        return
    logger.info(f"Loaded {len(documents_to_process)} new/modified documents.")

    # 5. Chunk Documents
    logger.info(f"Chunking documents using: {chunker.__class__.__name__}")
    all_chunks = []
    for doc in documents_to_process:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
    logger.info(f"Total number of chunks created: {len(all_chunks)}")

    # 6. Generate Embeddings
    logger.info(f"Generating embeddings using: {embedder.__class__.__name__}")
    chunk_contents = [chunk.content for chunk in all_chunks]
    embeddings = embedder.embed(chunk_contents)
    logger.debug(f"Embeddings generated with shape: {embeddings.shape}")

    # Assign embeddings back to each chunk's metadata
    for i, chunk in enumerate(all_chunks):
        chunk.metadata["embedding"] = embeddings[i]

    # 7. Sink Data
    logger.info(f"Sinking data to: {sink.__class__.__name__}")
    sink.sink(all_chunks)

    # 8. Update and Save State
    logger.info("Updating state for processed files...")
    for doc in documents_to_process:
        source_identifier = doc.metadata.get("source")
        if source_identifier:
            state_manager.update_state(source_identifier)
            logger.debug(f"Updated state for source: {source_identifier}")
    state_manager.save_state()

    logger.info("VectorFlow pipeline completed successfully.")
