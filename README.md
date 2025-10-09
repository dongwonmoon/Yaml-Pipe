# VectorFlow

> **Note:** This project is currently a work in progress. Features and configurations may change.

VectorFlow is a flexible ETL pipeline designed to streamline the process of converting text data into vector embeddings and loading them into a vector database. It allows you to define and run a multi-step pipeline using a simple YAML configuration file.

## Features

- **YAML-based Configuration**: Easily define your pipeline's components and their parameters in a `pipeline.yaml` file.
- **Pluggable Components**: Swap out components for different data sources, chunking strategies, embedding models, and data sinks.
- **Extensible**: Designed to be easily extended with new components.
- **Multiple Data Sources**: Load data from local files (`LocalFileSource`) or web pages (`WebSource`).
- **Vector Database Sink**: Currently supports sinking data into [LanceDB](https://lancedb.github.io/lancedb/).

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd vector-flow
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the embedding pipeline using the command-line interface.

1.  **Run the default pipeline:**

    This command uses the `pipeline.yaml` file, which reads from a local text file.

    ```bash
    python main.py run
    ```

2.  **Run a different pipeline:**

    Use the `--config-path` (or `-c`) option to specify a different configuration file. The `pipeline_web.yaml` file is provided as an example to process data from a URL.

    ```bash
    python main.py run --config-path pipeline_web.yaml
    ```

## Configuration

The pipeline is controlled by a YAML file (e.g., `pipeline.yaml`). It consists of four main sections: `source`, `chunker`, `embedder`, and `sink`.

```yaml
source:
  type: local_files
  config:
    path: ./data/vectorflow_intro.txt

chunker:
  type: recursive_character
  config:
    chunk_size: 150
    chunk_overlap: 30

embedder:
  type: sentence_transformer
  config:
    model_name: "jhgan/ko-sbert-nli"

sink:
  type: lancedb
  config:
    uri: "./lancedb_final"
    table_name: "my_documents"
```

- **`source`**: Defines where to get the data from.
  - `type`: `local_files` or `web`.
  - `config`: Parameters for the source type (e.g., `path` for files, `url` for web).
- **`chunker`**: Defines how to split the text into smaller pieces.
  - `type`: `recursive_character`.
  - `config`: Parameters for the chunker (e.g., `chunk_size`).
- **`embedder`**: Defines the model to use for creating vector embeddings.
  - `type`: `sentence_transformer`.
  - `config`: Parameters for the embedder (e.g., `model_name`).
- **`sink`**: Defines where to store the final text chunks and their embeddings.
  - `type`: `lancedb`.
  - `config`: Parameters for the sink (e.g., database `uri` and `table_name`).