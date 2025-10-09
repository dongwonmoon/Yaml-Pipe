"""
Core data models for the VectorFlow pipeline.

This module defines the standard data structures that are passed between
components in the pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Document:
    """
    A standard data packet that flows through the pipeline.

    This dataclass represents a piece of content, which could be a whole
    file or a smaller chunk. It holds the text content and associated
    metadata.

    Attributes:
        content (str): The text content of the document or chunk.
        metadata (Dict[str, Any]): A dictionary to hold metadata about the
            content. Examples include the source file path, URL, chunk index,
            or generated embeddings.
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
