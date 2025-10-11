"""
Tests for the data sink components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, ANY

from vectorflow.components.sinks import LanceDBSink, ChromaDBSink
from vectorflow.core.data_models import Document


@pytest.fixture
def sample_documents():
    """Fixture for a list of sample documents to be sinked."""
    return [
        Document(
            content="Doc 1",
            metadata={
                "source": "file1.txt",
                "embedding": np.array([0.1, 0.2]),
            },
        ),
        Document(
            content="Doc 2",
            metadata={
                "source": "file2.txt",
                "embedding": np.array([0.3, 0.4]),
            },
        ),
        Document(
            content="Doc 3 from file1",
            metadata={
                "source": "file1.txt",
                "embedding": np.array([0.5, 0.6]),
            },
        ),
    ]


# --- Tests for LanceDBSink ---


@patch("vectorflow.components.sinks.pydantic_to_schema")
@patch("vectorflow.components.sinks.create_dynamic_pydantic_model")
@patch("lancedb.connect")
def test_lancedb_sink_creates_table_if_not_exists(
    mock_connect, mock_create_model, mock_to_schema, sample_documents
):
    """Test that LanceDBSink creates a new table if it doesn't exist."""
    mock_db = MagicMock()
    mock_db.open_table.side_effect = ValueError
    mock_connect.return_value = mock_db

    sink = LanceDBSink(uri="/fake/db", table_name="test_table")
    sink.sink(sample_documents)

    mock_connect.assert_called_once_with("/fake/db")
    mock_db.create_table.assert_called_once_with("test_table", schema=ANY)
    mock_db.open_table.assert_called_once_with("test_table")
    mock_db.create_table.return_value.add.assert_called_once()


@patch("vectorflow.components.sinks.pydantic_to_schema")
@patch("vectorflow.components.sinks.create_dynamic_pydantic_model")
@patch("lancedb.connect")
def test_lancedb_sink_deletes_old_records(
    mock_connect, mock_create_model, mock_to_schema, sample_documents
):
    """Test that LanceDBSink deletes records from the same source before adding new ones."""
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_table.schema = MagicMock()
    mock_db.open_table.return_value = mock_table
    mock_connect.return_value = mock_db
    mock_to_schema.return_value = mock_table.schema  # Ensure schema matches

    sink = LanceDBSink(uri="/fake/db", table_name="test_table")
    sink.sink(sample_documents)

    # Check that delete was called with a WHERE clause for both sources
    mock_table.delete.assert_called_once()
    delete_clause = mock_table.delete.call_args[1]["where"]
    assert "source = 'file1.txt'" in delete_clause
    assert "source = 'file2.txt'" in delete_clause

    # Check that add was called with the new data
    mock_table.add.assert_called_once()
    added_data = mock_table.add.call_args[0][0]
    assert isinstance(added_data, pd.DataFrame)
    assert len(added_data) == 3


# --- Tests for ChromaDBSink ---


@patch("chromadb.PersistentClient")
def test_chromadb_sink_sinks_data(mock_persistent_client, sample_documents):
    """Test the basic sinking functionality of ChromaDBSink."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_persistent_client.return_value = mock_client
    mock_client.get_or_create_collection.return_value = mock_collection

    sink = ChromaDBSink(path="/fake/chroma", collection_name="test_collection")
    sink.sink(sample_documents)

    mock_client.get_or_create_collection.assert_called_once_with(
        name="test_collection"
    )

    # Check that delete was called for the correct sources
    mock_collection.delete.assert_called_once_with(
        where={"source": {"$in": ["file1.txt", "file2.txt"]}}
    )

    # Check that add was called with the correct data
    mock_collection.add.assert_called_once()
    add_args = mock_collection.add.call_args[1]
    assert len(add_args["ids"]) == 3
    assert len(add_args["documents"]) == 3
    assert len(add_args["metadatas"]) == 3
    assert len(add_args["embeddings"]) == 3
    assert add_args["documents"][0] == "Doc 1"
