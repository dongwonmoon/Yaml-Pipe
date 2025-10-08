from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Document:
    """A standard data packet for our pipeline."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
