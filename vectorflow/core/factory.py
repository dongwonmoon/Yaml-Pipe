from ..components.sources import LocalFileSource, WebSource
from ..components.chunkers import (
    RecursiveCharacterChunker,
    MarkdownChunker,
    AdaptiveChunker,
)
from ..components.embedders import SentenceTransformerEmbedder
from ..components.sinks import LanceDBSink, ChromaDBSink

SOURCE_REGISTRY = {"local_files": LocalFileSource, "web": WebSource}

CHUNKER_REGISTRY = {
    "recursive_character": RecursiveCharacterChunker,
    "markdown": MarkdownChunker,
    "adaptive": AdaptiveChunker,
}

EMBEDDER_REGISTRY = {
    "sentence_transformer": SentenceTransformerEmbedder,
}

SINK_REGISTRY = {"lancedb": LanceDBSink, "chromadb": ChromaDBSink}


def build_component(component_config: dict, registry: dict):
    """
    Builds a component instance from a configuration dictionary and a registry.

    Args:
        component_config: The component's configuration dictionary
                          (e.g., {'type': 'local_files', 'config': {'path': '...'}}).
        registry: The registry to use (e.g., SOURCE_REGISTRY).
    """
    component_type = component_config.get("type", "")
    config = component_config.get("config", {})

    if not component_type:
        raise ValueError("Component 'type' not specified in configuration.")

    component_class = registry.get(component_type)
    if not component_class:
        raise ValueError(f"'{component_type}' is not a valid component type.")

    return component_class(**config)
