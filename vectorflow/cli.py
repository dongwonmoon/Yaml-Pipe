"""
Command-Line Interface for VectorFlow.

This module provides the main entry point for the VectorFlow application,
using Typer to create a clean and user-friendly CLI.
"""

import typer
import logging
from pathlib import Path
import json
from typing_extensions import Annotated
import shutil

from .core.state_manager import StateManager
from .core.pipeline import run_pipeline
from .core.factory import (
    SOURCE_REGISTRY,
    SINK_REGISTRY,
    CHUNKER_REGISTRY,
    EMBEDDER_REGISTRY,
    build_component,
)
from .core.evaluation import Evaluator
from .utils.config import load_config


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
def list_components():
    """Lists all available components that can be used in the pipeline."""
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


@app.command(name="test-connection")
def test_connection(
    component: Annotated[
        str, typer.Argument(help="Component to test (e.g., 'source' or 'sink')")
    ],
    config_path: str = typer.Option(
        "pipeline.yaml", "-c", help="Path to the configuration file."
    ),
):
    """Tests the connection for a specified component based on the config file."""
    logger.info(f"Testing connection for component: '{component}'...")

    try:
        config = load_config(config_path)

        if component == "source":
            state_manager = StateManager()
            config["source"]["config"]["state_manager"] = state_manager
            comp_obj = build_component(config["source"], SOURCE_REGISTRY)
        elif component == "sink":
            comp_obj = build_component(config["sink"], SINK_REGISTRY)
        else:
            logger.error(f"Unknown component type: '{component}'")
            raise typer.Exit(code=1)

        comp_obj.test_connection()

    except Exception as e:
        logger.error(f"Error testing connection: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def clean(
    config_path: str = typer.Option("pipeline.yaml", "-c", help="Config file to use."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Removes all generated files, including the state file and sink database."""
    logger.info("Starting cleanup process...")

    if not yes:
        confirmed = typer.confirm("Are you sure you want to continue?")
        if not confirmed:
            logger.info("Aborting cleanup.")
            return

    state_file = Path(".vectorflow_state.json")
    if state_file.exists():
        logger.info(f"Removing state file: {state_file}")
        state_file.unlink()
        logger.info(f"Deleted state file: {state_file}")

    try:
        config = load_config(config_path)
        sink_config = config.get("sink", {}).get("config", {})

        sink_path_str = sink_config.get("uri") or sink_config.get("path")

        if sink_path_str:
            sink_path = Path(sink_path_str)
            if sink_path.exists() and sink_path.is_dir():
                shutil.rmtree(sink_path)
                logger.info(f"Deleted sink directory: {sink_path}")
    except SystemExit:
        logger.warning(
            f"Could not load config at '{config_path}' to clean Sink. Skipping."
        )

    logger.info("Cleanup process completed successfully.")


@app.command()
def eval(
    dataset_path: Annotated[
        str, typer.Argument(help="Path to the evaluation dataset (.jsonl file).")
    ],
    config_path: str = typer.Option(
        "pipeline.yaml",
        "-c",
        help="Path to the configuration file to use for the evaluation.",
    ),
    k: int = typer.Option(
        5, "--top-k", "-k", help="Number of top results to check for a hit."
    ),
):
    """
    Evaluates the performance of the vector database using a given dataset.
    """
    logger.info(f"Starting evaluation with config: '{config_path}'")

    try:
        config = load_config(config_path)

        embedder = build_component(config["embedder"], EMBEDDER_REGISTRY)
        sink_config = config["sink"]

        evaluator = Evaluator(embedder=embedder, sink_config=sink_config)

        evaluator.evaluate(dataset_path=dataset_path, k=k)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise typer.Exit(code=1)
