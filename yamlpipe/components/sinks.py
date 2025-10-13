"""
Data sink components for the YamlPipe pipeline.

This module provides classes for writing the final processed documents
(text chunks and their embeddings) to various destinations, such as
vector databases.
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
from chromadb.config import Settings


from ..utils.data_models import Document
from ..utils.dynamic_schemas import create_dynamic_pydantic_model

logger = logging.getLogger(__name__)


class BaseSink(ABC):
    """Abstract base class for all data sink components."""

    @abstractmethod
    def sink(self, documents: List[Document]):
        """
        Takes a list of processed documents and saves them to the destination.

        Args:
            documents (List[Document]): A list of Document objects, each containing
                                     a text chunk and its embedding metadata.
        """
        pass

    @abstractmethod
    def test_connection(self):
        """
        Tests the connection to the data sink to ensure it is accessible.

        Raises:
            Exception: If the connection test fails.
        """
        pass


class LanceDBSink(BaseSink):
    """
    A sink that writes data to a LanceDB table.

    This sink handles dynamic schema creation based on document metadata,
    table creation, and efficiently updates data by deleting records from
    changed sources before adding new ones.
    """

    def __init__(self, uri: str, table_name: str):
        """
        Initializes the LanceDBSink.

        Args:
            uri (str): The URI for the LanceDB database connection (e.g., a local path).
            table_name (str): The name of the table to sink data into.
        """
        self.uri = uri
        self.table_name = table_name
        logger.debug(f"Initialized LanceDBSink with uri='{uri}', table='{table_name}'")

    def _handle_schema_mismatch(self, db, table, new_schema):
        logger.warning(
            f"Schema mismatch for table '{self.table_name}'. Starting migration..."
        )

        temp_table_name = f"{self.table_name}_temp_{uuid.uuid4().hex[:6]}"
        logger.info(f"Creating temporary table: {temp_table_name}")
        temp_table = db.create_table(temp_table_name, schema=new_schema)

        logger.info(f"Migrating old data from '{self.table_name}'...")
        old_data = table.to_pandas()

        if not old_data.empty:
            migrated_df = pd.DataFrame(old_data).reindex(columns=new_schema.names)
            temp_table.add(migrated_df)
            logger.info(f"Migrated {len(old_data)} records to temporary table.")

        logger.info(f"Replacing old table with new one...")
        db.drop_table(self.table_name)
        new_table = db.create_table(self.table_name, schema=new_schema)
        if not old_data.empty:
            new_table.add(old_data)

        db.drop_table(temp_table_name, ignore_missing=True)

        logger.info("✅ Schema migration successful.")
        return new_table

    def sink(self, documents: List[Document]):
        """Sinks the given documents into a LanceDB table."""
        if not documents:
            logger.warning("No documents provided to sink. Aborting.")
            return

        logger.info(
            f"Sinking {len(documents)} documents to LanceDB table '{self.table_name}' at '{self.uri}'"
        )
        try:
            db = lancedb.connect(self.uri)
        except Exception as e:
            logger.error(
                f"Failed to connect to LanceDB at URI: '{self.uri}'",
                exc_info=True,
            )
            raise ConnectionError(f"Could not connect to LanceDB: {e}")

        try:
            DynamicModel = create_dynamic_pydantic_model(documents)
            pyarrow_schema = pydantic_to_schema(DynamicModel)
        except (ValueError, TypeError) as e:
            logger.error(
                f"Error creating dynamic Pydantic model or schema: {e}",
                exc_info=True,
            )
            return

        try:
            table = db.open_table(self.table_name)
            if table.schema != pyarrow_schema:
                table = self._handle_schema_mismatch(db, table, pyarrow_schema)
        except ValueError:
            if "migration required" in str(e).lower():
                raise
            logger.info(f"Table '{self.table_name}' not found. Creating new table.")
            table = db.create_table(self.table_name, schema=pyarrow_schema)

        docs_by_source = defaultdict(list)
        for doc in documents:
            if source := doc.metadata.get("source"):
                docs_by_source[source].append(doc)

        sources_to_delete = list(docs_by_source.keys())
        if sources_to_delete:
            where_clause = " OR ".join(
                [f"source = '{source}'" for source in sources_to_delete]
            )
            logger.info(f"Deleting existing records from sources: {sources_to_delete}")
            try:
                table.delete(where=where_clause)
            except Exception as e:
                logger.error(
                    f"Error deleting records: {e}. This may happen if the table is empty.",
                    exc_info=True,
                )

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

        if records:
            logger.info(
                f"Adding {len(records)} new records to table '{self.table_name}'."
            )
            table.add(pd.DataFrame(records))

        logger.info("Finished sinking data to LanceDB.")

    def test_connection(self):
        """Tests the connection to LanceDB by connecting and listing tables."""
        logger.info(f"Testing connection to LanceDB at URI: {self.uri}")
        try:
            db = lancedb.connect(self.uri)
            db.table_names()
            logger.info("Connection to LanceDB successful.")
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}", exc_info=True)
            raise ConnectionError(f"Failed to connect to LanceDB: {e}")


class ChromaDBSink(BaseSink):
    """A sink that writes data to a ChromaDB collection."""

    def __init__(self, host: str, port: int, collection_name: str):
        """
        Initializes the ChromaDBSink with server connection details.

        Args:
            host (str): The hostname of the ChromaDB server.
            port (int): The port of the ChromaDB server.
            collection_name (str): The name of the collection to sink data into.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = chromadb.HttpClient(host=self.host, port=self.port)
        logger.debug(
            f"Initialized ChromaDBSink with host='{host}', port='{port}', collection='{self.collection_name}'"
        )

    def sink(self, documents: List[Document]):
        """Sinks the given documents into a ChromaDB collection."""
        if not documents:
            logger.warning("No documents provided to sink. Aborting.")
            return

        logger.info(
            f"Sinking {len(documents)} documents to ChromaDB collection '{self.collection_name}' at {self.host}:{self.port}"
        )
        try:
            collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            logger.error(
                f"Failed to connect to ChromaDB server at {self.host}:{self.port}",
                exc_info=True,
            )
            raise ConnectionError(f"Could not connect to ChromaDB: {e}")

        docs_by_source = defaultdict(list)
        for doc in documents:
            if source := doc.metadata.get("source"):
                docs_by_source[source].append(doc)

        sources_to_delete = list(docs_by_source.keys())
        if sources_to_delete:
            logger.info(f"Deleting existing records from sources: {sources_to_delete}")
            try:
                collection.delete(where={"source": {"$in": sources_to_delete}})
            except Exception as e:
                logger.error(
                    f"Error deleting records from ChromaDB: {e}", exc_info=True
                )

        ids = [str(uuid.uuid4()) for _ in documents]
        contents = [doc.content for doc in documents]
        embeddings = [doc.metadata.get("embedding").tolist() for doc in documents]
        metadatas = []
        for doc in documents:
            clean_meta = doc.metadata.copy()
            clean_meta.pop("embedding")
            metadatas.append(clean_meta)

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
            logger.error(f"Error adding records to ChromaDB: {e}", exc_info=True)

    def test_connection(self):
        """Tests the connection to ChromaDB by connecting and counting collections."""
        logger.info(f"Testing connection for ChromaDBSink at {self.host}:{self.port}")
        try:
            self.client.heartbeat()
            logger.info("Connection to ChromaDB successful.")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}", exc_info=True)
            raise ConnectionError(f"Failed to connect to ChromaDB: {e}")
