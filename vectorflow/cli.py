import typer
from .pipeline import run_pipeline

app = typer.Typer()


@app.command()
def run(
    config_path: str = typer.Option(
        "pipeline.yml", "--config-path", "-c", help="사용할 파이프라인 설정 파일 경로"
    )
):
    """
    VectorFlow 임베딩 파이프라인을 실행합니다.
    """
    run_pipeline(config_path=config_path)
