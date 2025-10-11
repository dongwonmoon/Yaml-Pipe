"""
Component Factory for the VectorFlow pipeline.

This module implements the factory pattern for creating pipeline components.
It uses registries to map configuration strings (e.g., 'local_files') to
the actual component classes. This allows for a flexible and extensible
pipeline where components can be easily added or replaced via configuration.
"""

import logging
from ..components.sources import LocalFileSource, WebSource, S3Source
from ..components.chunkers import (
    RecursiveCharacterChunker,
    MarkdownChunker,
    AdaptiveChunker,
)
from ..components.embedders import SentenceTransformerEmbedder
from ..components.sinks import LanceDBSink, ChromaDBSink

logger = logging.getLogger(__name__)

# A registry mapping 'type' strings to their corresponding Source classes.
SOURCE_REGISTRY = {"local_files": LocalFileSource, "web": WebSource, "s3": S3Source}

# A registry mapping 'type' strings to their corresponding Chunker classes.
CHUNKER_REGISTRY = {
    "recursive_character": RecursiveCharacterChunker,
    "markdown": MarkdownChunker,
    "adaptive": AdaptiveChunker,
}

# A registry mapping 'type' strings to their corresponding Embedder classes.
EMBEDDER_REGISTRY = {
    "sentence_transformer": SentenceTransformerEmbedder,
}

# A registry mapping 'type' strings to their corresponding Sink classes.
SINK_REGISTRY = {"lancedb": LanceDBSink, "chromadb": ChromaDBSink}


def build_component(component_config: dict, registry: dict):
    """
    Builds a component instance from a configuration dictionary and a registry.

    This generic function takes a component's configuration, which must include
    a 'type' key, and looks up the corresponding class in the provided registry.
    It then instantiates the class with the parameters from the 'config' key.

    Args:
        component_config (dict): The component's configuration dictionary,
            expected to have 'type' and 'config' keys.
        registry (dict): The registry (e.g., SOURCE_REGISTRY) to look up the
            component class.

    Returns:
        An instance of the component class.

    Raises:
        ValueError: If the 'type' is not specified in the config or if the
            type is not found in the registry.
    """
    component_type = component_config.get("type", "")
    config = component_config.get("config", {})

    if not component_type:
        raise ValueError("Component 'type' not specified in configuration.")

    component_class = registry.get(component_type)
    if not component_class:
        raise ValueError(f"'{component_type}' is not a valid component type.")

    logger.debug(
        f"Building component '{component_class.__name__}' with config: {config}"
    )
    return component_class(**config)
