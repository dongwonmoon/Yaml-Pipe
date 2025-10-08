import typer
from .pipeline import run_pipeline

app = typer.Typer()


@app.command()
def run():
    print("CLI run 명령어가 호출되었습니다.")
    run_pipeline()
