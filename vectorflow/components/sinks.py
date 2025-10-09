
"""
Data sink components for the VectorFlow pipeline.

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

from ..core.data_models import Document
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


class LanceDBSink(BaseSink):
    """
    A sink that writes data to a LanceDB table.

    This sink handles schema creation, table creation, and efficiently updates
    data by deleting records from changed sources before adding new ones.
    """

    def __init__(self, uri: str, table_name: str):
        """
        Initializes the LanceDBSink.

        Args:
            uri (str): The URI for the LanceDB database connection.
            table_name (str): The name of the table to sink data into.
        """
        self.uri = uri
        self.table_name = table_name
        logger.debug(f"Initialized LanceDBSink with uri='{uri}', table='{table_name}'")

    def sink(self, documents: List[Document]):
        """Sinks the given documents into a LanceDB table."""
        logger.info(
            f"Sinking {len(documents)} documents to LanceDB table '{self.table_name}' at '{self.uri}'"
        )
        db = lancedb.connect(self.uri)

        if not documents:
            logger.warning("No documents provided to sink. Aborting.")
            return

        # Create a dynamic Pydantic model and PyArrow schema from the documents
        try:
            DynamicModel = create_dynamic_pydantic_model(documents)
            pyarrow_schema = pydantic_to_schema(DynamicModel)
        except (ValueError, TypeError) as e:
            logger.error(f"Error creating dynamic Pydantic model or schema: {e}", exc_info=True)
            return

        # Create or open the table, handling schema changes
        try:
            table = db.open_table(self.table_name)
            if table.schema != pyarrow_schema:
                logger.warning(
                    f"Schema mismatch for table '{self.table_name}'. Deleting and recreating table."
                )
                db.drop_table(self.table_name)
                table = db.create_table(self.table_name, schema=pyarrow_schema)
        except FileNotFoundError:
            logger.info(f"Table '{self.table_name}' not found. Creating new table.")
            table = db.create_table(self.table_name, schema=pyarrow_schema)

        # Group documents by their source for efficient deletion
        docs_by_source = defaultdict(list)
        for doc in documents:
            if source := doc.metadata.get("source"):
                docs_by_source[source].append(doc)

        # Delete existing records from the table that match the sources of the new documents
        sources_to_delete = list(docs_by_source.keys())
        if sources_to_delete:
            where_clause = " OR ".join([f"source = '{source}'" for source in sources_to_delete])
            logger.info(f"Deleting existing records from sources: {sources_to_delete}")
            try:
                table.delete(where=where_clause)
                logger.debug(f"Deletion successful for sources: {sources_to_delete}")
            except Exception as e:
                logger.error(f"Error deleting records: {e}", exc_info=True)

        # Prepare records for insertion
        records = []
        for doc in documents:
            record = {
                "text": doc.content,
                "vector": doc.metadata.get("embedding"),
            }
            # Add all other metadata fields, ensuring they are not nested dicts
            for key, value in doc.metadata.items():
                if key != "embedding":
                    record[key] = value
            records.append(record)

        # Add the new records to the table
        if records:
            data = pd.DataFrame(records)
            logger.info(f"Adding {len(data)} new records to table '{self.table_name}'.")
            table.add(data)

        logger.info("Finished sinking data to LanceDB.")


class ChromaDBSink(BaseSink):
    """A sink that writes data to a ChromaDB collection."""

    def __init__(self, path: str, collection_name: str):
        """
        Initializes the ChromaDBSink.

        Args:
            path (str): The path to the directory where ChromaDB should store its data.
            collection_name (str): The name of the collection to sink data into.
        """
        self.path = path
        self.collection_name = collection_name
        logger.debug(
            f"Initialized ChromaDBSink with path='{path}', collection='{collection_name}'"
        )

    def sink(self, documents: List[Document]):
        """Sinks the given documents into a ChromaDB collection."""
        logger.info(
            f"Sinking {len(documents)} documents to ChromaDB collection '{self.collection_name}' at '{self.path}'"
        )
        client = chromadb.PersistentClient(path=self.path)
        collection = client.get_or_create_collection(name=self.collection_name)

        if not documents:
            logger.warning("No documents provided to sink. Aborting.")
            return

        # Group documents by their source for efficient deletion
        docs_by_source = defaultdict(list)
        for doc in documents:
            if source := doc.metadata.get("source"):
                docs_by_source[source].append(doc)

        # Delete existing records from the collection that match the sources of the new documents
        sources_to_delete = list(docs_by_source.keys())
        if sources_to_delete:
            logger.info(f"Deleting existing records from sources: {sources_to_delete}")
            collection.delete(where={"source": {"$in": sources_to_delete}})

        # Prepare records for insertion
        ids = [str(uuid.uuid4()) for _ in documents]
        contents = [doc.content for doc in documents]
        embeddings = [doc.metadata.get("embedding").tolist() for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        logger.info(f"Adding {len(documents)} new records to collection '{self.collection_name}'.")
        collection.add(
            ids=ids, documents=contents, metadatas=metadatas, embeddings=embeddings
        )
        logger.info("Finished sinking data to ChromaDB.")
