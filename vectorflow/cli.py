import typer
import logging
from pathlib import Path
import json
from .pipeline import run_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = typer.Typer()


@app.command()
def run(
    config_path: str = typer.Option(
        "pipeline.yaml",
        "--config-path",
        "-c",
        help="Path to the pipeline configuration file to use.",
    )
):
    """Runs the VectorFlow embedding pipeline."""
    logger.info("VectorFlow pipeline starting.")
    run_pipeline(config_path=config_path)


@app.command()
def init():
    """
    Create new VectorFlow Project in current directory.
    """
    logger.info("Create new VectorFlow Project.")
    Path("data").mkdir(exist_ok=True)

    config_file = Path("pipeline.yaml")
    if config_file.exists():
        logger.info("'pipeline.yaml' already exists.")
    else:
        DEFAULT_YAML_CONTENT = """
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

        config_file.write_text(DEFAULT_YAML_CONTENT)
        logger.info("'pipeline.yaml' created.")

    logger.info("VectorFlow Project created successfully.")


@app.command()
def status():
    state_file = Path(".vectorflow_state.json")
    if not state_file.exists():
        logger.info("No VectorFlow Project found.")
        return

    with open(state_file, "r") as f:
        state = json.load(f)

    processed_files = state.get("processed_files", {})
    if not processed_files:
        logger.info("No files processed yet.")
    else:
        for file_path in processed_files.keys():
            print(f"  - {file_path}")
