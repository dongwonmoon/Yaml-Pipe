# YamlPipe

YamlPipe is a flexible ETL pipeline designed to streamline the process of converting text data into vector embeddings and loading them into a vector database. It allows you to define and run a multi-step pipeline using a simple YAML configuration file.

## Features

- **YAML-based Configuration**: Easily define your pipeline's components and their parameters in a `pipeline.yaml` file.
- **Pluggable Components**: Swap out components for different data sources, chunking strategies, embedding models, and data sinks.
- **Extensible**: Designed to be easily extended with new components.
- **Multiple Data Sources**: Load data from local files (`local_files`), web pages (`web`), S3 buckets (`s3`), and PostgreSQL databases (`postgres`).
- **Advanced Chunking**: Choose from `recursive_character`, `markdown`, or `adaptive` chunking strategies.
- **Multiple Embedding Models**: Use `sentence_transformer` or `openai` models.
- **Multiple Vector Databases**: Sink data into `lancedb` or `chromadb`.
- **CLI**: A powerful CLI to run pipelines, manage projects, and test components.
- **Web UI**: A Streamlit-based dashboard to run pipelines and test search.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/vector-flow.git
    cd vector-flow
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  To use the Web UI, install the UI-specific dependencies:
    ```bash
    pip install -r requirements-ui.txt
    ```

## Usage

### Command-Line Interface

YamlPipe provides a powerful CLI for managing your projects.

- **Initialize a new project:**
  ```bash
  python main.py init
  ```

- **Run a pipeline:**
  ```bash
  python main.py run -c pipelines/pipeline.yaml
  ```

- **List available components:**
  ```bash
  python main.py list-components
  ```

- **Check the status of processed files:**
  ```bash
  python main.py status
  ```

- **Test the connection to a source or sink:**
  ```bash
  python main.py test-connection source -c pipelines/pipeline.yaml
  ```

- **Clean up generated files:**
  ```bash
  python main.py clean -c pipelines/pipeline.yaml --yes
  ```

- **Evaluate the pipeline:**
  ```bash
  python main.py eval eval_dataset.jsonl -c pipelines/pipeline.yaml
  ```

### Web Interface

Run the Streamlit web interface for a more interactive experience.

```bash
streamlit run app.py
```

## Configuration

The pipeline is controlled by a YAML file. Here's an example with all available components:

```yaml
source:
  type: local_files
  config:
    path: ./data
    glob_pattern: "*.txt"

chunker:
  type: adaptive
  config:
    chunk_size: 200
    chunk_overlap: 40

embedder:
  type: sentence_transformer
  config:
    model_name: "jhgan/ko-sbert-nli"

sink:
  type: chromadb
  config:
    path: "./chroma_data"
    collection_name: "my_documents"
```

- **`source`**: `local_files`, `web`, `s3`, `postgres`
- **`chunker`**: `recursive_character`, `markdown`, `adaptive`
- **`embedder`**: `sentence_transformer`, `openai`
- **`sink`**: `lancedb`, `chromadb`