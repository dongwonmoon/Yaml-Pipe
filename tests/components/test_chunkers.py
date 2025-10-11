from yamlpipe.utils.data_models import Document
from yamlpipe.components.chunkers import RecursiveCharacterChunker


def test_recursive_character_chunker():
    """Tests the basic functionality of the RecursiveCharacterChunker."""
    chunker = RecursiveCharacterChunker(chunk_size=30, chunk_overlap=5)
    doc = Document(
        content="This is a test sentence for our amazing chunker.",
        metadata={"source": "test.txt"},
    )

    chunked_docs = chunker.chunk(doc)

    assert len(chunked_docs) > 1
    assert chunked_docs[0].content == "This is a test sentence for"
    assert chunked_docs[1].content == "for our amazing chunker."

    assert chunked_docs[0].metadata["source"] == "test.txt"
    assert chunked_docs[0].metadata["chunk_index"] == 1
    assert chunked_docs[1].metadata["chunk_index"] == 2
