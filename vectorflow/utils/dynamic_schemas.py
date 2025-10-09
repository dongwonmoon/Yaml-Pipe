from pydantic import BaseModel, create_model
from typing import List, Dict, Any, Type
from lancedb.pydantic import Vector
from ..core.data_models import Document

TYPE_MAP = {
    str: (str, ...),
    int: (int, ...),
    float: (float, ...),
}


def create_dynamic_pydantic_model(documents: List[Document]) -> Type[BaseModel]:
    """
    Dynamically generate Pydantic models by analyzing a list of documents.
    """
    if not documents:
        raise ValueError("At least one document is required to create a schema.")

    metadata_fields = {}
    for doc in documents:
        for key, value in doc.metadata.items():
            if key not in metadata_fields and type(value) in TYPE_MAP:
                metadata_fields[key] = type(value)

    pydantic_fields = {}
    vector_dim = documents[0].metadata["embedding"].shape[0]
    pydantic_fields["text"] = (str, ...)
    pydantic_fields["vector"] = (Vector(vector_dim), ...)

    for key, value_type in metadata_fields.items():
        if key != "embedding":
            pydantic_fields[key] = TYPE_MAP[value_type]

    DynamicDocumentModel = create_model("DynamicDocumentModel", **pydantic_fields)

    return DynamicDocumentModel
