from pydantic import BaseModel
from typing import Dict, Any


class ComponentConfig(BaseModel):
    """A model for a single component's configuration (source, chunker, etc.)"""

    type: str
    config: Dict[str, Any] = {}


class PipelineConfig(BaseModel):
    """The top-level model for the entire pipeline.yml configuration."""

    source: ComponentConfig
    chunker: ComponentConfig
    embedder: ComponentConfig
    sink: ComponentConfig
