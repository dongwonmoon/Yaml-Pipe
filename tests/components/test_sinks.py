"""
Tests for the data sink components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, ANY

from yamlpipe.components.sinks import LanceDBSink, ChromaDBSink
from yamlpipe.utils.data_models import Document


@pytest.fixture
def sample_documents():
    """Provides a list of sample Document objects for testing sinks."""
    return [
        Document(
            content="Doc 1",
            metadata={"source": "file1.txt", "embedding": np.array([0.1, 0.2])},
        ),
        Document(
            content="Doc 2",
            metadata={"source": "file2.txt", "embedding": np.array([0.3, 0.4])},
        ),
    ]


@patch("yamlpipe.components.sinks.pydantic_to_schema")
@patch("yamlpipe.components.sinks.create_dynamic_pydantic_model")
@patch("lancedb.connect")
def test_lancedb_sink(
    mock_connect, mock_create_model, mock_to_schema, sample_documents
):
    """Tests the basic functionality of the LanceDBSink."""
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_table.schema = MagicMock()
    mock_db.open_table.return_value = mock_table
    mock_connect.return_value = mock_db
    mock_to_schema.return_value = mock_table.schema

    sink = LanceDBSink(uri="/fake/db", table_name="test_table")
    sink.sink(sample_documents)

    mock_connect.assert_called_with("/fake/db")
    mock_db.open_table.assert_called_with("test_table")
    mock_table.delete.assert_called_once()
    mock_table.add.assert_called_once()


@pytest.mark.parametrize(
    "client_type, sink_params",
    [
        ("PersistentClient", {"path": "/fake/chroma"}),
        ("HttpClient", {"host": "localhost", "port": 8000}),
    ],
)
@patch("chromadb.PersistentClient")
@patch("chromadb.HttpClient")
def test_chromadb_sink(
    mock_http_client,
    mock_persistent_client,
    client_type,
    sink_params,
    sample_documents,
):
    """Tests the ChromaDBSink with both PersistentClient and HttpClient."""
    mock_client = MagicMock()
    mock_collection = MagicMock()

    if client_type == "PersistentClient":
        mock_persistent_client.return_value = mock_client
    else:
        mock_http_client.return_value = mock_client

    mock_client.get_or_create_collection.return_value = mock_collection

    sink = ChromaDBSink(collection_name="test_collection", **sink_params)
    sink.sink(sample_documents)

    mock_client.get_or_create_collection.assert_called_with(
        name="test_collection"
    )
    mock_collection.delete.assert_called_once()
    mock_collection.add.assert_called_once()
    add_args = mock_collection.add.call_args[1]
    assert len(add_args["ids"]) == 2
    assert add_args["documents"][0] == "Doc 1"
