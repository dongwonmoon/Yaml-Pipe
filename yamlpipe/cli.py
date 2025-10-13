"""
Command-Line Interface for YamlPipe.
"""

import typer
import logging
from pathlib import Path
import json
from typing_extensions import Annotated
import shutil

from .utils.state_manager import StateManager
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


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="A flexible ETL pipeline for vector embeddings.")


@app.command()
def run(
    config_path: str = typer.Option(
        "pipelines/pipeline.yaml",
        "-c",
        help="Path to the pipeline's YAML configuration file.",
    )
):
    """Runs the YamlPipe embedding pipeline."""
    run_pipeline(config_path=config_path)


@app.command()
def init():
    """Initializes a new YamlPipe project."""
    logger.info("Initializing new YamlPipe project...")
    Path("data").mkdir(exist_ok=True)
    logger.info("Created 'data' directory.")

    config_file = Path("pipeline.yaml")
    if config_file.exists():
        logger.warning("'pipeline.yaml' already exists.")
    else:
        DEFAULT_YAML_CONTENT = """# Default YamlPipe Pipeline Configuration
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
    model_name: "jhgan/ko-sbert-nli"

sink:
  type: lancedb
  config:
    uri: "./lancedb_data"
    table_name: "documents"
"""
        config_file.write_text(DEFAULT_YAML_CONTENT.strip())
        logger.info("Created default 'pipeline.yaml'.")

    logger.info("Project initialized.")


@app.command()
def status():
    """Shows the status of the YamlPipe project."""
    state_file = Path(".yamlpipe_state.json")
    if not state_file.exists():
        logger.warning("No state file found. Run a pipeline first.")
        return

    try:
        with open(state_file, "r") as f:
            state = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error reading state file: {e}", exc_info=True)
        return

    processed_items = state.get("processed_items", {})
    if not processed_items:
        logger.info("No items have been processed yet.")
    else:
        print("\n--- Tracked Items ---")
        for item_id in sorted(processed_items.keys()):
            print(f"  - {item_id}")
        print("---------------------")


@app.command(name="list-components")
def list_components():
    """Lists all available components."""
    logger.info("Listing available components...")

    def print_registry(title, registry):
        print(f"\n--- {title} ---")
        for name in sorted(registry.keys()):
            print(f"  - {name}")

    print_registry("Sources", SOURCE_REGISTRY)
    print_registry("Chunkers", CHUNKER_REGISTRY)
    print_registry("Embedders", EMBEDDER_REGISTRY)
    print_registry("Sinks", SINK_REGISTRY)


@app.command(name="test-connection")
def test_connection(
    component: Annotated[
        str, typer.Argument(help="Component to test (source or sink)")
    ],
    config_path: str = typer.Option("pipeline.yaml", "-c", help="Config path."),
):
    """Tests the connection for a specified component."""
    logger.info(f"Testing connection for '{component}'...")
    try:
        config = load_config(config_path)
        if component == "source":
            state_manager = StateManager()
            config["source"]["config"]["state_manager"] = state_manager
            comp_obj = build_component(config["source"], SOURCE_REGISTRY)
        elif component == "sink":
            comp_obj = build_component(config["sink"], SINK_REGISTRY)
        else:
            logger.error(f"Unknown component: '{component}'")
            raise typer.Exit(code=1)
        comp_obj.test_connection()
    except Exception as e:
        logger.error(f"Connection test failed: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def clean(
    config_path: str = typer.Option("pipeline.yaml", "-c", help="Config file."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
):
    """Removes all generated files."""
    logger.info("Starting cleanup...")
    if not yes and not typer.confirm("Are you sure?"):
        logger.info("Aborting cleanup.")
        return

    state_file = Path(".yamlpipe_state.json")
    if state_file.exists():
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
    except Exception as e:
        logger.warning(f"Could not clean sink: {e}", exc_info=True)

    logger.info("Cleanup complete.")


@app.command()
def eval(
    dataset_path: Annotated[
        str, typer.Argument(help="Path to evaluation dataset.")
    ],
    config_path: str = typer.Option("pipeline.yaml", "-c", help="Config path."),
    k: int = typer.Option(5, "--top-k", "-k", help="Top k results to check."),
):
    """Evaluates the vector database performance."""
    logger.info(f"Starting evaluation with config: '{config_path}'")
    try:
        config = load_config(config_path)
        embedder = build_component(config["embedder"], EMBEDDER_REGISTRY)
        evaluator = Evaluator(embedder=embedder, sink_config=config["sink"])
        evaluator.evaluate(dataset_path=dataset_path, k=k)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise typer.Exit(code=1)
