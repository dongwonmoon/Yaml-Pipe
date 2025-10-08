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
from .dynamic_schemas import create_dynamic_pydantic_model


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

        try:
            DynamicModel = create_dynamic_pydantic_model(documents)
            pyarrow_schema = pydantic_to_schema(DynamicModel)
        except ValueError as e:
            logger.error(f"Error creating dynamic Pydantic model: {e}")
            return

        records = []
        for doc in documents:
            record = {
                "text": doc.content,
                "vector": doc.metadata.get("embedding"),
            }
            for key, value in doc.metadata.items():
                if key != "embedding":
                    record[key] = value
            records.append(record)

        data = pd.DataFrame(records)

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
