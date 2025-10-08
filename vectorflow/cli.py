import typer
import logging
from .pipeline import run_pipeline

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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("VectorFlow pipeline starting.")
    run_pipeline(config_path=config_path)
