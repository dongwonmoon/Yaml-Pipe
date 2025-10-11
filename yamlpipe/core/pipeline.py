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


def _build_components(config: dict, state_manager: StateManager) -> tuple:
    """Builds all pipeline components based on the configuration."""
    logger.info("Building pipeline components...")
    try:
        # Inject the state manager into the source component's configuration
        config["source"]["config"]["state_manager"] = state_manager
        source = build_component(config["source"], SOURCE_REGISTRY)
        chunker = build_component(config["chunker"], CHUNKER_REGISTRY)
        embedder = build_component(config["embedder"], EMBEDDER_REGISTRY)
        sink = build_component(config["sink"], SINK_REGISTRY)
        logger.info("All components built successfully.")
        return source, chunker, embedder, sink
    except (ValueError, KeyError) as e:
        logger.error(f"Error building components: {e}", exc_info=True)
        raise


def _process_documents(source, chunker, embedder, sink, state_manager, config):
    """Loads, processes, and sinks the documents."""
    logger.info(f"Loading data from source: {source.__class__.__name__}")
    documents_to_process = source.load_data()

    if not documents_to_process:
        logger.info("No new or modified documents to process. Pipeline finished.")
        return

    logger.info(f"Loaded {len(documents_to_process)} new/modified documents.")

    logger.info(f"Chunking documents using: {chunker.__class__.__name__}")
    all_chunks = [chunk for doc in documents_to_process for chunk in chunker.chunk(doc)]
    logger.info(f"Total number of chunks created: {len(all_chunks)}")

    if not all_chunks:
        logger.info(
            "No chunks were created from the documents. Nothing to embed or sink."
        )
        return

    logger.info(f"Generating embeddings using: {embedder.__class__.__name__}")
    chunk_contents = [chunk.content for chunk in all_chunks]
    embeddings = embedder.embed(chunk_contents)
    logger.debug(f"Embeddings generated with shape: {embeddings.shape}")

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["embedding"] = embeddings[i]

    logger.info(f"Sinking data to: {sink.__class__.__name__}")
    sink.sink(all_chunks)

    logger.info("Updating state for processed files...")
    source_type = config.get("source", {}).get("type")
    if source_type == "local_files" or source_type == "s3":
        for doc in documents_to_process:
            source_identifier = doc.metadata.get("source")
            if source_identifier:
                state_manager.update_state(source_identifier)
                logger.debug(f"Updated state for source: {source_identifier}")
    elif source_type == "postgres":
        state_manager.update_run_timestamp()
    state_manager.save_state()


def run_pipeline(config_path: str):
    """
    Runs the entire embedding pipeline based on a configuration file.
    """
    logger.info(f"YamlPipe pipeline starting with config: {config_path}")

    try:
        state_manager = StateManager()
        config = load_config(config_path)
        if not config:
            logger.error(
                "Configuration is empty or could not be loaded. Aborting pipeline."
            )
            return

        source, chunker, embedder, sink = _build_components(config, state_manager)
        _process_documents(source, chunker, embedder, sink, state_manager, config)

        logger.info("YamlPipe pipeline completed successfully.")

    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
