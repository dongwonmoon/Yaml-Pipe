"""
Tests for the data source components.
"""

import pytest
from unittest.mock import patch, MagicMock
import requests

from yamlpipe.components.sources import LocalFileSource, WebSource
from yamlpipe.utils.data_models import Document


@pytest.fixture
def mock_state_manager():
    """Fixture to create a mock StateManager."""
    manager = MagicMock()
    # Default behavior: has_changed returns True for any file
    manager.has_changed.return_value = True
    return manager


# --- Tests for LocalFileSource ---


def test_local_source_handles_empty_dir(mock_state_manager):
    """Test that LocalFileSource returns an empty list for an empty directory."""
    with patch("pathlib.Path.glob") as mock_glob:
        mock_glob.return_value = []

        source = LocalFileSource(
            path="/fake/data",
            glob_pattern="*.txt",
            state_manager=mock_state_manager,
        )
        documents = source.load_data()

        assert len(documents) == 0


# --- Tests for WebSource ---


@patch("requests.get")
def test_web_source_loads_data_successfully(mock_requests_get):
    """Test that WebSource successfully fetches and parses a web page."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body><h1>Title</h1><p>Hello world</p></body></html>"
    mock_requests_get.return_value = mock_response

    source = WebSource(url="http://fake-url.com")
    documents = source.load_data()

    mock_requests_get.assert_called_once_with("http://fake-url.com", timeout=10)
    assert len(documents) == 1
    assert "Hello world" in documents[0].content
    assert documents[0].metadata["source"] == "http://fake-url.com"


@patch("requests.get")
def test_web_source_handles_request_exception(mock_requests_get):
    """Test that WebSource returns an empty list when a request fails."""
    mock_requests_get.side_effect = requests.exceptions.RequestException("Test error")

    source = WebSource(url="http://fake-url.com")
    documents = source.load_data()

    assert len(documents) == 0


@patch("requests.head")
def test_web_source_connection_test_success(mock_requests_head):
    """Test a successful connection test for WebSource."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_requests_head.return_value = mock_response

    source = WebSource(url="http://fake-url.com")
    source.test_connection()  # Should not raise an exception


@patch("requests.head")
def test_web_source_connection_test_failure(mock_requests_head):
    """Test a failed connection test for WebSource."""
    mock_requests_head.side_effect = requests.exceptions.RequestException(
        "Connection failed"
    )

    source = WebSource(url="http://fake-url.com")
    with pytest.raises(ConnectionError):
        source.test_connection()
