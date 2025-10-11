import pytest
from vectorflow.core.factory import (
    build_component,
    SOURCE_REGISTRY,
    CHUNKER_REGISTRY,
)
from vectorflow.components.sources import LocalFileSource
from vectorflow.components.chunkers import RecursiveCharacterChunker
from vectorflow.core.state_manager import StateManager


def test_build_source_component():
    """Tests if the factory correctly builds a source component."""
    config = {
        "type": "local_files",
        "config": {
            "path": "./data",
            "glob_pattern": "*.txt",
            "state_manager": StateManager(),  # Mock state manager
        },
    }
    source_component = build_component(config, SOURCE_REGISTRY)
    assert isinstance(source_component, LocalFileSource)
    assert source_component.path.name == "data"


def test_build_chunker_component():
    """Tests if the factory correctly builds a chunker component with config."""
    config = {
        "type": "recursive_character",
        "config": {"chunk_size": 123, "chunk_overlap": 45},
    }
    chunker_component = build_component(config, CHUNKER_REGISTRY)
    assert isinstance(chunker_component, RecursiveCharacterChunker)
    assert chunker_component.chunk_size == 123
    assert chunker_component.chunk_overlap == 45


def test_build_component_invalid_type():
    """Tests if the factory raises an error for an unknown type."""
    config = {"type": "non_existent_type", "config": {}}
    with pytest.raises(
        ValueError, match="'non_existent_type' is not a valid component type."
    ):
        build_component(config, SOURCE_REGISTRY)
