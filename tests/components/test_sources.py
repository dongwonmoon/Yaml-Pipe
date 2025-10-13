"""
Tests for the data source components.
"""

import pytest
from unittest.mock import patch, MagicMock
import requests
import psycopg2

from yamlpipe.components.sources import (
    LocalFileSource,
    WebSource,
    S3Source,
    PostgreSQLSource,
)
from yamlpipe.utils.data_models import Document


@pytest.fixture
def mock_state_manager():
    """Provides a mock StateManager that reports no changes."""
    manager = MagicMock()
    manager.has_changed.return_value = True
    return manager


@patch("pathlib.Path.glob")
@patch(
    "unstructured.partition.auto.partition",
    return_value=[MagicMock(text="test content")],
)
def test_local_source_loads_data(mock_partition, mock_glob, mock_state_manager):
    """Tests that LocalFileSource can load data from the filesystem."""
    mock_file = MagicMock()
    mock_file.is_file.return_value = True
    mock_glob.return_value = [mock_file]

    source = LocalFileSource(
        path="/fake/data",
        glob_pattern="*.txt",
        state_manager=mock_state_manager,
    )
    documents = source.load_data()

    assert len(documents) == 1
    assert documents[0].content == "test content"


@patch("requests.get")
def test_web_source_loads_data(mock_requests_get):
    """Tests that WebSource can fetch and parse a web page."""
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Hello world</p></body></html>"
    mock_requests_get.return_value = mock_response

    source = WebSource(url="http://fake-url.com")
    documents = source.load_data()

    assert len(documents) == 1
    assert "Hello world" in documents[0].content


@patch("boto3.client")
def test_s3_source_loads_data(mock_boto3_client, mock_state_manager):
    """Tests that S3Source can load data from an S3 bucket."""
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3
    mock_s3.list_objects_v2.return_value = {
        "Contents": [{"Key": "test.txt", "ETag": "123"}]
    }
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=MagicMock(return_value=b"test content"))
    }

    source = S3Source(
        bucket="test-bucket", prefix="", state_manager=mock_state_manager
    )
    documents = source.load_data()

    assert len(documents) == 1
    assert documents[0].content == "test content"


@patch("psycopg2.connect")
def test_postgres_source_loads_data(mock_psycopg2_connect, mock_state_manager):
    """Tests that PostgreSQLSource can load data from a database."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_psycopg2_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchall.return_value = [{"content": "test content", "id": 1}]

    source = PostgreSQLSource(
        host="localhost",
        port=5432,
        database="testdb",
        user="user",
        password="pass",
        query="SELECT * FROM test",
        state_manager=mock_state_manager,
    )
    documents = source.load_data()

    assert len(documents) == 1
    assert documents[0].content == "test content"
