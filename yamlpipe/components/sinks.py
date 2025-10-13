"""
Data sink components for the YamlPipe pipeline.
"""

from abc import ABC, abstractmethod
import pandas as pd
import logging
import uuid
from typing import List

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

    def _handle_schema_mismatch(self, db, table, new_schema):
        """Handles schema migration by recreating the table."""
        logger.warning(
            f"Schema mismatch for table '{self.table_name}'. Migrating..."
        )
        old_data = table.to_pandas()
        db.drop_table(self.table_name)
        new_table = db.create_table(self.table_name, schema=new_schema)
        if not old_data.empty:
            new_table.add(old_data)
        return new_table

    def sink(self, documents: List[Document]):
        if not documents:
            return

        db = lancedb.connect(self.uri)
        DynamicModel = create_dynamic_pydantic_model(documents)
        pyarrow_schema = pydantic_to_schema(DynamicModel)

        try:
            table = db.open_table(self.table_name)
            if table.schema != pyarrow_schema:
                table = self._handle_schema_mismatch(db, table, pyarrow_schema)
        except (FileNotFoundError, ValueError):
            table = db.create_table(self.table_name, schema=pyarrow_schema)

        # Delete existing records from the same sources to prevent duplicates.
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
            try:
                table.delete(where=where_clause)
            except Exception as e:
                logger.warning(f"Could not delete records: {e}")

        # Prepare records for insertion.
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
            table.add(pd.DataFrame(records))

    def test_connection(self):
        try:
            db = lancedb.connect(self.uri)
            db.table_names()
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
        # Support both on-disk and remote ChromaDB.
        if path:
            self.client = chromadb.PersistentClient(path=path)
        elif host and port:
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            raise ValueError(
                "Either 'path' or 'host' and 'port' must be provided."
            )

    def sink(self, documents: List[Document]):
        if not documents:
            return

        collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        # Delete existing records from the same sources.
        sources_to_delete = list(
            set(
                doc.metadata.get("source")
                for doc in documents
                if doc.metadata.get("source")
            )
        )
        if sources_to_delete:
            try:
                collection.delete(where={"source": {"$in": sources_to_delete}})
            except Exception as e:
                logger.warning(f"Could not delete records: {e}")

        # Prepare records for insertion.
        ids = [str(uuid.uuid4()) for _ in documents]
        contents = [doc.content for doc in documents]
        embeddings = [
            doc.metadata.get("embedding").tolist() for doc in documents
        ]
        metadatas = [doc.metadata.copy() for doc in documents]
        for meta in metadatas:
            meta.pop("embedding", None)

        if contents:
            collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

    def test_connection(self):
        try:
            self.client.heartbeat()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ChromaDB: {e}") from e
