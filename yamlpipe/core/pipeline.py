"""
Core pipeline orchestration module.

This module defines the main function `run_pipeline` that reads a YAML configuration,
builds the necessary components (source, chunker, embedder, sink), and executes
the ETL pipeline to process text data into vector embeddings.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from ..utils.config import load_config
from .factory import (
    build_component,
    SOURCE_REGISTRY,
    CHUNKER_REGISTRY,
    EMBEDDER_REGISTRY,
    SINK_REGISTRY,
)
from ..utils.state_manager import StateManager
from ..utils.data_models import Document

logger = logging.getLogger(__name__)


def _process_document_chunk(doc, chunker):
    """Chunks a single document and returns the chunks."""
    try:
        return chunker.chunk(doc)
    except Exception as e:
        logger.error(f"Error chunking document: {e}", exc_info=True)
        return []


def _build_components(config: dict, state_manager: StateManager) -> tuple:
    """Builds all pipeline components based on the configuration."""
    logger.info("Building pipeline components...")
    try:
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
    max_workers = min(4, os.cpu_count() or 1)
    logger.info(f"Using {max_workers} workers for parallel processing.")

    logger.info(f"Loading data from source: {source.__class__.__name__}")
    documents_to_process = source.load_data()

    if not documents_to_process:
        logger.info("No new or modified documents to process. Pipeline finished.")
        return

    logger.info(f"Loaded {len(documents_to_process)} new/modified documents.")

    logger.info(f"Chunking documents using: {chunker.__class__.__name__}")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_document_chunk, doc, chunker): doc
            for doc in documents_to_process
        }
        all_chunks = []
        processed_docs = []
        logger.info("Processing chunks...")
        for future in as_completed(futures):
            result_chunks = future.result()
            if result_chunks:
                all_chunks.extend(result_chunks)
                processed_docs.append(futures[future])

    logger.info(f"Total number of chunks created: {len(all_chunks)}")

    if not all_chunks:
        logger.info("No chunks were created. Nothing to embed or sink.")
        return

    logger.info(f"Generating embeddings using: {embedder.__class__.__name__}")
    chunk_contents = [chunk.content for chunk in all_chunks]
    embeddings = embedder.embed(chunk_contents)

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["embedding"] = embeddings[i]

    logger.info(f"Sinking data to: {sink.__class__.__name__}")
    sink.sink(all_chunks)

    logger.info("Updating state for processed files...")
    source.update_state(processed_docs)
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
            logger.error("Configuration is empty. Aborting pipeline.")
            return

        source, chunker, embedder, sink = _build_components(config, state_manager)
        _process_documents(source, chunker, embedder, sink, state_manager, config)

        logger.info("YamlPipe pipeline completed successfully.")

    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
