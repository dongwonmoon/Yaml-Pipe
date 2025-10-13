"""
Utility for dynamically creating Pydantic models for LanceDB.
"""

import logging
from pydantic import BaseModel, create_model
from typing import List, Type
import numpy as np
import datetime

from lancedb.pydantic import Vector
from .data_models import Document

logger = logging.getLogger(__name__)

TYPE_MAP = {
    str: (str, ...),
    int: (int, ...),
    float: (float, ...),
    list: (List, ...),
    datetime.datetime: (datetime.datetime, ...),
}


def create_dynamic_pydantic_model(documents: List[Document]) -> Type[BaseModel]:
    """
    Dynamically generates a Pydantic model from a list of documents.
    """
    if not documents:
        raise ValueError(
            "At least one document is required to create a schema."
        )

    logger.debug("Starting dynamic Pydantic model creation.")

    metadata_fields = {}
    for doc in documents:
        for key, value in doc.metadata.items():
            if key not in metadata_fields and type(value) in TYPE_MAP:
                metadata_fields[key] = type(value)
                logger.debug(
                    f"Discovered metadata field '{key}' with type {type(value)}."
                )

    pydantic_fields = {}
    pydantic_fields["text"] = (str, ...)

    first_embedding = documents[0].metadata.get("embedding")
    if first_embedding is None or not isinstance(first_embedding, np.ndarray):
        raise ValueError("First document must have a valid numpy embedding.")
    vector_dim = first_embedding.shape[0]
    pydantic_fields["vector"] = (Vector(vector_dim), ...)
    logger.debug(f"Inferred vector dimension: {vector_dim}")

    for key, value_type in metadata_fields.items():
        if key != "embedding":
            if value_type in TYPE_MAP:
                pydantic_fields[key] = TYPE_MAP[value_type]
            else:
                raise TypeError(
                    f"Unsupported type '{value_type}' for metadata key '{key}'."
                )

    DynamicDocumentModel = create_model(
        "DynamicDocumentModel", **pydantic_fields
    )
    logger.debug(
        f"Created DynamicDocumentModel with fields: {list(pydantic_fields.keys())}"
    )

    return DynamicDocumentModel
