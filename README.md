<p align="center">
  <img src="https://raw.githubusercontent.com/dongwonmoon/Yaml-Pipe/main/assets/banner.png" width="480" alt="YamlPipe Logo">
</p>

<h1 align="center">🧩 YamlPipe</h1>

<p align="center">
  A lightweight, YAML-driven <b>ETL pipeline</b> that transforms text data into vector embeddings — <br>
  with zero boilerplate, full flexibility, and seamless database integration.
</p>

<p align="center">
  <a href="https://github.com/dongwonmoon/Yaml-Pipe/stargazers"><img src="https://img.shields.io/github/stars/dongwonmoon/Yaml-Pipe?style=social" alt="GitHub Stars"></a>
  <a href="https://github.com/dongwonmoon/Yaml-Pipe/blob/main/LICENSE"><img src="https://img.shields.io/github/license/dongwonmoon/Yaml-Pipe" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.11+-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/YAML-Pipeline-orange" alt="YAML Based">
</p>

---

## 🚀 Overview

**YamlPipe** lets you build end-to-end ETL pipelines for vector embedding workflows — all defined in a single YAML file.

It’s designed for **AI developers**, **data engineers**, and **RAG (Retrieval-Augmented Generation)** builders who want simplicity without losing flexibility.

With YamlPipe, you can:

- ✅ Load data from files, web, S3, or Postgres  
- 🧠 Chunk text dynamically using multiple strategies  
- ⚙️ Generate embeddings with OpenAI or Sentence Transformers  
- 🧩 Store vectors in LanceDB or ChromaDB  
- 💻 Run everything via CLI or Web UI (Streamlit)

---

## 🧠 Features

- **YAML-based Configuration** – define your pipeline once, run it anywhere  
- **Pluggable Components** – modular architecture for each stage  
- **Advanced Chunking** – `recursive_character`, `markdown`, or `adaptive`  
- **Multiple Embedding Models** – `sentence_transformer` and `openai`  
- **Vector Database Integration** – `lancedb` or `chromadb`  
- **CLI & Streamlit UI** – full control, both terminal and browser  

---

## ⚡ Installation

```bash
git clone https://github.com/dongwonmoon/Yaml-Pipe.git
cd Yaml-Pipe
pip install -r requirements.txt
```

---

## 🧩 Quick Start

```bash
python main.py init
python main.py run -c pipelines/pipeline.yaml
```

### Example Pipeline
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

---

## 🌐 Web Interface

```bash
streamlit run app.py
```

Use the dashboard to visualize your pipelines, test search results, and monitor ingestion progress.

---

## 💡 Why YamlPipe?

- No more boilerplate ETL code — define everything in YAML  
- Designed for RAG, embedding pipelines, and AI data workflows  
- Fully open-source and easily extendable  

---

## 🧭 Roadmap

- [ ] Add Milvus / Pinecone sinks  
- [ ] Support LangChain / LlamaIndex integrations  
- [ ] Add benchmarking and pipeline visualization  

---

## 🤝 Contributing

Contributions are always welcome!  
Fork the repo, create a feature branch, and submit a PR.  
New ideas, documentation improvements, and bug reports are all appreciated.

---

## ⭐ Support

If YamlPipe helps you, please consider giving it a **star** 🌟  
Every star motivates continued development and new features!

<p align="center">
  <a href="https://github.com/dongwonmoon/Yaml-Pipe/stargazers">
    <img src="https://img.shields.io/github/stars/dongwonmoon/Yaml-Pipe?style=social" alt="Star YamlPipe">
  </a>
</p>

---

## 🪪 License

MIT © [dongwonmoon](https://github.com/dongwonmoon)
