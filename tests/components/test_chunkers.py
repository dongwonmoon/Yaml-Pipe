import pytest
from yamlpipe.utils.data_models import Document
from yamlpipe.components.chunkers import (
    RecursiveCharacterChunker,
    MarkdownChunker,
    AdaptiveChunker,
)


@pytest.fixture
def sample_document():
    return Document(
        content="This is a test sentence for our amazing chunker. It is a long sentence.",
        metadata={"source": "test.txt"},
    )


def test_recursive_character_chunker(sample_document):
    """Tests the RecursiveCharacterChunker."""
    chunker = RecursiveCharacterChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.chunk(sample_document)
    assert len(chunks) > 1
    assert chunks[0].content == "This is a test sentence for"
    assert chunks[1].content == "for our amazing chunker. It"
    assert chunks[0].metadata["source"] == "test.txt"


def test_markdown_chunker():
    """Tests the MarkdownChunker."""
    doc = Document(
        content="# Header 1\n\nThis is a paragraph.\n\n## Header 2\n\n- List item 1\n- List item 2",
        metadata={"source": "test.md"},
    )
    chunker = MarkdownChunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert chunks[0].content.startswith("# Header 1")
    assert chunks[1].content.startswith("## Header 2")


def test_adaptive_chunker(sample_document):
    """Tests the AdaptiveChunker."""
    chunker = AdaptiveChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.chunk(sample_document)
    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "test.txt"
