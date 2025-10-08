from abc import ABC, abstractmethod
import pandas as pd
import logging
from pydantic import BaseModel
import uuid

import lancedb
from lancedb.pydantic import pydantic_to_schema, Vector
import chromadb


logger = logging.getLogger(__name__)


class BaseSink(ABC):
    @abstractmethod
    def sink(self, data: pd.DataFrame):
        """
        data: {"vector": [], "text": []}
        """
        pass


class LanceDBSink(BaseSink):
    def __init__(self, uri: str, table_name: str):
        self.uri = uri
        self.table_name = table_name

    def sink(self, data: pd.DataFrame):
        """Sinks the given DataFrame into a LanceDB table."""
        logger.info(
            f"Sinking data to LanceDB. URI: {self.uri}, Table: {self.table_name}"
        )
        self.db = lancedb.connect(self.uri)

        vector_dimensions = data["vector"].iloc[0].shape[0]

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

    def sink(self, data: pd.DataFrame):
        """Sinks the given DataFrame into a ChromaDB collection."""
        logger.info(
            f"Sinking data to ChromaDB. Path: {self.path}, Collection: {self.collection_name}"
        )
        client = chromadb.PersistentClient(path=self.path)
        collection = client.get_or_create_collection(name=self.collection_name)

        embeddings = data["vector"].tolist()
        documents = data["text"].tolist()
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids,
        )
        logger.info("Finished sinking data to vector DB.")
