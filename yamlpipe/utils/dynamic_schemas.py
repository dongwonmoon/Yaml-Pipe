"""
Utility for dynamically creating Pydantic models for LanceDB.

This module provides a function to generate a Pydantic model on-the-fly
based on the contents of a list of Document objects. This is necessary
to create a LanceDB schema that matches the specific metadata fields present
in the data being processed.
"""

import logging
from pydantic import BaseModel, create_model
from typing import List, Type, get_args, get_origin
import numpy as np
import datetime

from lancedb.pydantic import Vector
from .data_models import Document

logger = logging.getLogger(__name__)

# Maps Python types to Pydantic field definitions. The `...` indicates a required field.
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

    The model is created by inspecting the metadata of the first document to
    determine the vector dimension and analyzing all documents to find all unique
    metadata keys and their types. It includes 'text' and 'vector' fields by
    default.

    Args:
        documents (List[Document]): A list of Document objects to inspect.

    Returns:
        Type[BaseModel]: A dynamically created Pydantic BaseModel class.

    Raises:
        ValueError: If the document list is empty or if the first document
                    lacks an embedding.
        TypeError: If a metadata value has a type not supported by the TYPE_MAP.
    """
    if not documents:
        raise ValueError(
            "At least one document is required to create a schema."
        )

    logger.debug("Starting dynamic Pydantic model creation.")

    # Infer metadata fields and their types from all documents
    metadata_fields = {}
    for doc in documents:
        for key, value in doc.metadata.items():
            if key not in metadata_fields and type(value) in TYPE_MAP:
                metadata_fields[key] = type(value)
                logger.debug(
                    f"Discovered metadata field '{key}' with type {type(value)}."
                )

    # Step 1: Inspect all documents to find all unique metadata keys and their types.
    # This ensures the schema can accommodate all data variations.
    metadata_fields = {}
    for doc in documents:
        for key, value in doc.metadata.items():
            if key not in metadata_fields and type(value) in TYPE_MAP:
                metadata_fields[key] = type(value)
                logger.debug(
                    f"Discovered metadata field '{key}' with type {type(value)}."
                )

    # Step 2: Define the fields for the Pydantic model, starting with defaults.
    pydantic_fields = {}

    # Add the mandatory 'text' field for the document content.
    pydantic_fields["text"] = (str, ...)

    # Step 3: Infer the vector dimension from the first document's embedding.
    # This is a critical step as all vectors in a LanceDB table must have the same dimension.
    first_embedding = documents[0].metadata.get("embedding")
    if first_embedding is None or not isinstance(first_embedding, np.ndarray):
        raise ValueError(
            "First document must have a valid numpy embedding to infer vector dimension."
        )
    vector_dim = first_embedding.shape[0]
    pydantic_fields["vector"] = (Vector(vector_dim), ...)
    logger.debug(f"Inferred vector dimension: {vector_dim}")

    # Step 4: Add fields for all other metadata keys discovered in Step 1.
    # The raw 'embedding' is excluded as it's already handled by the 'vector' field.
    for key, value_type in metadata_fields.items():
        if (
            key != "embedding"
        ):  # Avoid adding the raw embedding as a separate field
            if value_type in TYPE_MAP:
                pydantic_fields[key] = TYPE_MAP[value_type]
            else:
                # This case should ideally not be hit due to the check during discovery
                raise TypeError(
                    f"Unsupported type '{value_type}' for metadata key '{key}'."
                )

    # Step 5: Create the Pydantic model class dynamically using the collected fields.
    DynamicDocumentModel = create_model(
        "DynamicDocumentModel", **pydantic_fields
    )
    logger.debug(
        f"Created DynamicDocumentModel with fields: {list(pydantic_fields.keys())}"
    )

    return DynamicDocumentModel
