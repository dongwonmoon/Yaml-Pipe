from abc import ABC, abstractmethod
import pandas as pd
import logging
from pydantic import BaseModel
import uuid
from typing import List
from collections import defaultdict

import lancedb
from lancedb.pydantic import pydantic_to_schema, Vector
import chromadb

from ..core.data_models import Document
from ..utils.dynamic_schemas import create_dynamic_pydantic_model


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

        try:
            table = self.db.open_table(self.table_name)
            if table.schema != pyarrow_schema:
                logger.info("The schema has changed, so the table will be regenerated.")
                self.db.drop_table(self.table_name)
                table = self.db.create_table(self.table_name, schema=pyarrow_schema)
        except FileNotFoundError:
            logger.info("The table does not exist, so create a new one.")
            table = self.db.create_table(self.table_name, schema=pyarrow_schema)

        docs_by_source = defaultdict(list)
        for doc in documents:
            source = doc.metadata.get("source")
            if source:
                docs_by_source[doc.metadata["source"]].append(doc)

        sources_to_delete = list(docs_by_source.keys())
        if sources_to_delete:
            where_clause = " OR ".join(
                [f"source == '{source}'" for source in sources_to_delete]
            )
            logger.info(f"Delete exist data. Target source: {sources_to_delete}")
            table.delete(where=where_clause)

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
        logger.info(f"Add new {len(data)} records.")

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

        docs_by_source = defaultdict(int)
        for doc in documents:
            source = doc.metadata.get("source")
            if source:
                docs_by_source[source].append(doc)

        sources_to_delete = list(docs_by_source.keys())
        if sources_to_delete:
            logger.info(f"Delete exist data. Target source: {sources_to_delete}")
            collection.delete(where={"source": {"$in": sources_to_delete}})

        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        contents = [doc.content for doc in documents]
        embeddings = [doc.metadata.get("embedding").tolist() for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        collection.add(
            ids=ids, documents=contents, metadatas=metadatas, embeddings=embeddings
        )
        logger.info("Finished sinking data to vector DB.")
