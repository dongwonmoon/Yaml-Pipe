import pytest
from yamlpipe.core.factory import (
    build_component,
    SOURCE_REGISTRY,
    CHUNKER_REGISTRY,
    EMBEDDER_REGISTRY,
    SINK_REGISTRY,
)
from yamlpipe.components.sources import LocalFileSource
from yamlpipe.components.chunkers import RecursiveCharacterChunker
from yamlpipe.components.embedders import SentenceTransformerEmbedder
from yamlpipe.components.sinks import LanceDBSink
from yamlpipe.utils.state_manager import StateManager, JSONStateManager


def test_build_source_component():
    """Tests if the factory correctly builds a source component."""
    config = {
        "type": "local_files",
        "config": {
            "path": "./data",
            "glob_pattern": "*.txt",
            "state_manager": StateManager(backend=JSONStateManager()),
        },
    }
    component = build_component(config, SOURCE_REGISTRY)
    assert isinstance(component, LocalFileSource)


def test_build_chunker_component():
    """Tests if the factory correctly builds a chunker component."""
    config = {
        "type": "recursive_character",
        "config": {"chunk_size": 100, "chunk_overlap": 10},
    }
    component = build_component(config, CHUNKER_REGISTRY)
    assert isinstance(component, RecursiveCharacterChunker)


def test_build_embedder_component():
    """Tests if the factory correctly builds an embedder component."""
    config = {
        "type": "sentence_transformer",
        "config": {"model_name": "test-model"},
    }
    component = build_component(config, EMBEDDER_REGISTRY)
    assert isinstance(component, SentenceTransformerEmbedder)


def test_build_sink_component():
    """Tests if the factory correctly builds a sink component."""
    config = {
        "type": "lancedb",
        "config": {"uri": "/tmp/lancedb", "table_name": "test"},
    }
    component = build_component(config, SINK_REGISTRY)
    assert isinstance(component, LanceDBSink)


def test_build_component_invalid_type():
    """Tests if the factory raises a ValueError for an invalid component type."""
    config = {"type": "invalid_type", "config": {}}
    with pytest.raises(ValueError):
        build_component(config, SOURCE_REGISTRY)
