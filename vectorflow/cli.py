"""
Command-Line Interface for VectorFlow.

This module provides the main entry point for the VectorFlow application,
using Typer to create a clean and user-friendly CLI.
"""

import typer
import logging
from pathlib import Path
import json
from .core.pipeline import run_pipeline
from .core.factory import (
    SOURCE_REGISTRY,
    SINK_REGISTRY,
    CHUNKER_REGISTRY,
    EMBEDDER_REGISTRY,
)

# Configure logging for clear, user-friendly output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create a Typer app instance, which helps in creating the CLI commands
app = typer.Typer(help="A flexible ETL pipeline for vector embeddings.")


@app.command()
def run(
    config_path: str = typer.Option(
        "pipeline.yaml",
        "--config-path",
        "-c",
        help="Path to the pipeline's YAML configuration file.",
    )
):
    """
    Runs the VectorFlow embedding pipeline using a specified configuration file.
    """
    run_pipeline(config_path=config_path)


@app.command()
def init():
    """
    Initializes a new VectorFlow project in the current directory.

    This command creates a 'data' directory for source files and a default
    'pipeline.yaml' configuration file to get started quickly.
    """
    logger.info("Initializing new VectorFlow project...")

    # Create a directory for source data
    Path("data").mkdir(exist_ok=True)
    logger.info("Created 'data' directory.")

    # Create a default pipeline.yaml if it doesn't exist
    config_file = Path("pipeline.yaml")
    if config_file.exists():
        logger.warning("'pipeline.yaml' already exists. Skipping creation.")
    else:
        DEFAULT_YAML_CONTENT = """# Default VectorFlow Pipeline Configuration

source:
  type: local_files
  config:
    path: ./data
    glob_pattern: "*.txt"

chunker:
  type: recursive_character
  config:
    chunk_size: 200
    chunk_overlap: 40

embedder:
  type: sentence_transformer
  config:
    # Model for Korean language
    model_name: "jhgan/ko-sbert-nli"

sink:
  type: lancedb
  config:
    uri: "./lancedb_data"
    table_name: "documents"
"""
        config_file.write_text(DEFAULT_YAML_CONTENT.strip())
        logger.info("Created default 'pipeline.yaml'.")

    logger.info("VectorFlow project initialized successfully.")


@app.command()
def status():
    """
    Shows the status of the VectorFlow project by listing processed files.

    This command reads the .vectorflow_state.json file and displays a list
    of all file sources that have been successfully processed and are being tracked.
    """
    logger.info("Checking project status...")
    state_file = Path(".vectorflow_state.json")

    if not state_file.exists():
        logger.warning("No state file found. Run a pipeline first to generate state.")
        return

    try:
        with open(state_file, "r") as f:
            state = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error reading state file '{state_file}': {e}", exc_info=True)
        return

    processed_files = state.get("processed_files", {})
    if not processed_files:
        logger.info("State is empty. No files have been processed yet.")
    else:
        print("\n--- Tracked Files ---")
        for file_path in sorted(processed_files.keys()):
            print(f"  - {file_path}")
        print("---------------------")


@app.command(name="list-components")
def list_componenets():
    """ """
    logger.info("Listing available components...")

    def print_registry(title, registry):
        print(f"\n--- {title} ---")
        if not registry:
            print("No components available.")
            return
        for name in sorted(registry.keys()):
            print(f"  - {name}")

    print_registry("Sources", SOURCE_REGISTRY)
    print_registry("Chunkers", CHUNKER_REGISTRY)
    print_registry("Embedders", EMBEDDER_REGISTRY)
    print_registry("Sinks", SINK_REGISTRY)
