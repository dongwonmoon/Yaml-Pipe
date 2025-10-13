"""
Data sink components for the YamlPipe pipeline.
"""

from abc import ABC, abstractmethod
import pandas as pd
import logging
import uuid
from typing import List
from collections import defaultdict

import lancedb
from lancedb.pydantic import pydantic_to_schema
import chromadb

from ..utils.data_models import Document
from ..utils.dynamic_schemas import create_dynamic_pydantic_model

logger = logging.getLogger(__name__)


class BaseSink(ABC):
    """Abstract base class for all data sink components."""

    @abstractmethod
    def sink(self, documents: List[Document]):
        pass

    @abstractmethod
    def test_connection(self):
        pass


class LanceDBSink(BaseSink):
    """A sink that writes data to a LanceDB table."""

    def __init__(self, uri: str, table_name: str):
        self.uri = uri
        self.table_name = table_name
        logger.debug(
            f"Initialized LanceDBSink with uri='{uri}', table='{table_name}'"
        )

    def _handle_schema_mismatch(self, db, table, new_schema):
        logger.warning(
            f"Schema mismatch for table '{self.table_name}'. Migrating..."
        )
        old_data = table.to_pandas()
        logger.info(f"Backing up {len(old_data)} records.")
        db.drop_table(self.table_name)
        logger.info("Old table dropped.")
        new_table = db.create_table(self.table_name, schema=new_schema)
        if not old_data.empty:
            new_table.add(old_data)
            logger.info("Data migrated to new schema.")
        return new_table

    def sink(self, documents: List[Document]):
        if not documents:
            logger.warning("No documents to sink.")
            return

        logger.info(
            f"Sinking {len(documents)} documents to LanceDB table '{self.table_name}' at '{self.uri}'"
        )
        try:
            db = lancedb.connect(self.uri)
        except Exception as e:
            raise ConnectionError(f"Could not connect to LanceDB: {e}") from e

        DynamicModel = create_dynamic_pydantic_model(documents)
        pyarrow_schema = pydantic_to_schema(DynamicModel)

        try:
            table = db.open_table(self.table_name)
            if table.schema != pyarrow_schema:
                table = self._handle_schema_mismatch(db, table, pyarrow_schema)
        except (FileNotFoundError, ValueError):
            logger.info(
                f"Table '{self.table_name}' not found. Creating new table."
            )
            table = db.create_table(self.table_name, schema=pyarrow_schema)

        sources_to_delete = list(
            set(
                doc.metadata.get("source")
                for doc in documents
                if doc.metadata.get("source")
            )
        )
        if sources_to_delete:
            where_clause = " OR ".join(
                [f"source = '{source}'" for source in sources_to_delete]
            )
            logger.info(
                f"Deleting existing records from sources: {sources_to_delete}"
            )
            try:
                table.delete(where=where_clause)
            except Exception as e:
                logger.warning(
                    f"Could not delete records: {e}. This might be okay if the table was empty."
                )

        records = []
        for doc in documents:
            record = {
                "text": doc.content,
                "vector": doc.metadata.get("embedding"),
            }
            record.update(
                {k: v for k, v in doc.metadata.items() if k != "embedding"}
            )
            records.append(record)

        if records:
            logger.info(
                f"Adding {len(records)} new records to table '{self.table_name}'."
            )
            table.add(pd.DataFrame(records))

        logger.info("Finished sinking data to LanceDB.")

    def test_connection(self):
        logger.info(f"Testing connection to LanceDB at URI: {self.uri}")
        try:
            db = lancedb.connect(self.uri)
            db.table_names()
            logger.info("Connection to LanceDB successful.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to LanceDB: {e}") from e


class ChromaDBSink(BaseSink):
    """A sink that writes data to a ChromaDB collection."""

    def __init__(
        self,
        collection_name: str,
        path: str = None,
        host: str = None,
        port: int = None,
    ):
        self.collection_name = collection_name
        if path:
            self.client = chromadb.PersistentClient(path=path)
        elif host and port:
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            raise ValueError(
                "Either 'path' or both 'host' and 'port' must be provided for ChromaDBSink"
            )

    def sink(self, documents: List[Document]):
        if not documents:
            logger.warning("No documents to sink.")
            return

        logger.info(
            f"Sinking {len(documents)} documents to ChromaDB collection '{self.collection_name}'"
        )
        collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        sources_to_delete = list(
            set(
                doc.metadata.get("source")
                for doc in documents
                if doc.metadata.get("source")
            )
        )
        if sources_to_delete:
            logger.info(
                f"Deleting existing records from sources: {sources_to_delete}"
            )
            try:
                collection.delete(where={"source": {"$in": sources_to_delete}})
            except Exception as e:
                logger.warning(
                    f"Could not delete records: {e}. This might be okay if the collection was empty."
                )

        ids = [str(uuid.uuid4()) for _ in documents]
        contents = [doc.content for doc in documents]
        embeddings = [
            doc.metadata.get("embedding").tolist() for doc in documents
        ]
        metadatas = [doc.metadata.copy() for doc in documents]
        for meta in metadatas:
            meta.pop("embedding", None)

        logger.info(
            f"Adding {len(documents)} new records to collection '{self.collection_name}'."
        )
        try:
            collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            logger.info("Finished sinking data to ChromaDB.")
        except Exception as e:
            logger.error(
                f"Error adding records to ChromaDB: {e}", exc_info=True
            )

    def test_connection(self):
        logger.info("Testing connection to ChromaDB")
        try:
            self.client.heartbeat()
            logger.info("Connection to ChromaDB successful.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ChromaDB: {e}") from e
