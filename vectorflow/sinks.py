from abc import ABC, abstractmethod
import pandas as pd
import logging
from pydantic import BaseModel
import uuid
from typing import List

import lancedb
from lancedb.pydantic import pydantic_to_schema, Vector
import chromadb

from .data_models import Document


logger = logging.getLogger(__name__)


class BaseSink(ABC):
    @abstractmethod
    def sink(self, documents: List[Document]):
        """
        Takes a list of final chunk Documents and saves them to the destination.
        """
        pass


class LanceDBSink(BaseSink):
    def __init__(self, uri: str, table_name: str):
        self.uri = uri
        self.table_name = table_name

    def sink(self, documents: List[Document]):
        """Sinks the given DataFrame into a LanceDB table."""
        logger.info(
            f"Sinking data to LanceDB. URI: {self.uri}, Table: {self.table_name}"
        )
        self.db = lancedb.connect(self.uri)

        texts = [doc.content for doc in documents]
        vectors = [doc.metadata.get("embedding") for doc in documents]

        data_df = pd.DataFrame({"text": texts, "vector": vectors})
        vector_dimensions = data_df["vector"].iloc[0].shape[0]

        class Document(BaseModel):
            text: str
            vector: Vector(vector_dimensions)

        pyarrow_schema = pydantic_to_schema(Document)

        self.db.drop_table(self.table_name, ignore_missing=True)
        table = self.db.create_table(self.table_name, schema=pyarrow_schema)

        table.add(data)
        logger.info("Finished sinking data to vector DB.")


class ChromaDBSink(BaseSink):
    def __init__(self, path: str, collection_name: str):
        self.path = path
        self.collection_name = collection_name

    def sink(self, documents: List[Document]):
        """Sinks the given DataFrame into a ChromaDB collection."""
        logger.info(
            f"Sinking data to ChromaDB. Path: {self.path}, Collection: {self.collection_name}"
        )
        client = chromadb.PersistentClient(path=self.path)
        collection = client.get_or_create_collection(name=self.collection_name)

        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        contents = [doc.content for doc in documents]
        embeddings = [doc.metadata.get("embedding").tolist() for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        collection.add(
            ids=ids, documents=contents, metadatas=metadatas, embeddings=embeddings
        )
        logger.info("Finished sinking data to vector DB.")
