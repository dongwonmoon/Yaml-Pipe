"""
Data source components for the YamlPipe pipeline.

This module contains classes for loading data from various sources,
such as local files and web pages. Each source is responsible for
fetching raw data and converting it into a list of Document objects.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import logging
from typing import List
from unstructured.partition.auto import partition
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import psycopg2
from psycopg2.extras import DictCursor

from ..utils.state_manager import StateManager
from ..utils.data_models import Document

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    """Abstract base class for all data source components."""

    @abstractmethod
    def load_data(self) -> List[Document]:
        """
        Loads data from the configured source and returns a list of Documents.
        """
        pass

    @abstractmethod
    def update_state(self, processed_docs: List[Document]):
        """
        Updates the state of the source after processing.
        """
        pass

    @abstractmethod
    def test_connection(self):
        """
        Tests the connection to the data source to ensure it is accessible.
        """
        pass


class LocalFileSource(BaseSource):
    """
    Loads documents from the local filesystem.
    """

    def __init__(
        self,
        path: str,
        glob_pattern: str,
        state_manager: StateManager,
    ):
        self.path = Path(path)
        self.glob_pattern = glob_pattern
        self.state_manager = state_manager
        logger.debug(
            f"Initialized LocalFileSource with path='{self.path}' and glob='{self.glob_pattern}'"
        )

    def load_data(self) -> List[Document]:
        logger.info(
            f"Scanning for files in '{self.path}' with pattern '{self.glob_pattern}'."
        )
        if not self.path.is_dir():
            logger.error(f"Source path '{self.path}' is not a valid directory.")
            return []

        all_files = [
            str(f) for f in self.path.glob(self.glob_pattern) if f.is_file()
        ]
        new_or_changed_files = [
            f for f in all_files if self.state_manager.has_changed(f)
        ]

        if not new_or_changed_files:
            logger.info("No new or changed files detected.")
            return []

        logger.info(f"Found {len(new_or_changed_files)} new or changed files.")

        loaded_data = []
        for file_path in new_or_changed_files:
            try:
                elements = partition(filename=file_path)
                content = "\n\n".join([str(el) for el in elements])
                if not content.strip():
                    logger.warning(f"File '{file_path}' is empty. Skipping.")
                    continue
                doc = Document(content=content, metadata={"source": file_path})
                loaded_data.append(doc)
            except Exception as e:
                logger.error(
                    f"Error processing file '{file_path}': {e}", exc_info=True
                )
        return loaded_data

    def update_state(self, processed_docs: List[Document]):
        for doc in processed_docs:
            source_identifier = doc.metadata.get("source")
            if source_identifier:
                self.state_manager.update_file_state(source_identifier)

    def test_connection(self):
        logger.info(
            f"Testing connection for LocalFileSource at path: {self.path}"
        )
        if not self.path.exists():
            raise FileNotFoundError(
                f"Source path '{self.path}' does not exist."
            )
        if not self.path.is_dir():
            raise NotADirectoryError(
                f"Source path '{self.path}' is not a directory."
            )
        logger.info("Connection to LocalFileSource successful.")


class WebSource(BaseSource):
    """
    Loads a document from a web URL.
    """

    def __init__(self, url: str, **kwargs):
        self.url = url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

    def load_data(self) -> List[Document]:
        logger.info(f"Fetching content from URL: {self.url}")
        try:
            response = requests.get(self.url, timeout=10, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            clean_text = "\n".join(line for line in lines if line)
            if not clean_text.strip():
                logger.warning(f"No text content found at URL: {self.url}")
                return []
            return [Document(content=clean_text, metadata={"source": self.url})]
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Failed to fetch content from URL '{self.url}': {e}",
                exc_info=True,
            )
            return []

    def update_state(self, processed_docs: List[Document]):
        pass  # WebSource is stateless

    def test_connection(self):
        logger.info(f"Testing connection for WebSource at URL: {self.url}")
        try:
            response = requests.head(self.url, timeout=5)
            response.raise_for_status()
            logger.info("Connection to WebSource successful.")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to URL: {self.url}"
            ) from e


class S3Source(BaseSource):
    """
    Loads documents from an AWS S3 bucket.
    """

    def __init__(self, bucket: str, prefix: str, state_manager: StateManager):
        self.bucket_name = bucket
        self.prefix = prefix
        self.state_manager = state_manager
        self.s3_client = boto3.client("s3")

    def load_data(self) -> List[Document]:
        logger.info(f"Loading data from S3 bucket: {self.bucket_name}")
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=self.prefix
            )
            all_objects = response.get("Contents", [])
        except ClientError as e:
            logger.error(
                f"Error listing objects in S3 bucket: {e}", exc_info=True
            )
            return []

        new_or_changed_objects = []
        for obj in all_objects:
            source_id = f"s3://{self.bucket_name}/{obj['Key']}"
            if self.state_manager.has_changed(
                source_id, obj["ETag"].strip("'")
            ):
                new_or_changed_objects.append(obj)

        if not new_or_changed_objects:
            logger.info("No new or changed objects detected in S3.")
            return []

        logger.info(
            f"Found {len(new_or_changed_objects)} new or changed objects."
        )

        loaded_documents = []
        for obj in new_or_changed_objects:
            obj_key = obj["Key"]
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name, Key=obj_key
                )
                content = response["Body"].read().decode("utf-8")
                doc = Document(
                    content=content,
                    metadata={
                        "source": f"s3://{self.bucket_name}/{obj_key}",
                        "etag": obj["ETag"].strip("'"),
                    },
                )
                loaded_documents.append(doc)
            except Exception as e:
                logger.error(
                    f"Error loading object {obj_key}: {e}", exc_info=True
                )

        return loaded_documents

    def update_state(self, processed_docs: List[Document]):
        for doc in processed_docs:
            source_id = doc.metadata.get("source")
            etag = doc.metadata.get("etag")
            if source_id and etag:
                self.state_manager.update_file_state(source_id, etag)

    def test_connection(self):
        logger.info(f"Testing connection to S3 bucket: {self.bucket_name}")
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info("Connection to S3 bucket successful.")
        except NoCredentialsError as e:
            raise Exception("AWS credentials not found.") from e
        except ClientError as e:
            raise ConnectionError(
                f"Failed to connect to S3 bucket: {self.bucket_name}"
            ) from e


class PostgreSQLSource(BaseSource):
    """
    Loads data from a PostgreSQL database.
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        query: str,
        state_manager: StateManager,
        timestamp_column: str = "updated_at",
    ):
        self.db_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }
        self.query = query
        self.state_manager = state_manager
        self.timestamp_column = timestamp_column

    def load_data(self) -> List[Document]:
        logger.info("Loading data from PostgreSQL database")
        last_run_ts = self.state_manager.get_last_run_timestamp()
        final_query = self.query
        if last_run_ts:
            if "where" in self.query.lower():
                final_query += f" AND {self.timestamp_column} > '{last_run_ts}'"
            else:
                final_query += (
                    f" WHERE {self.timestamp_column} > '{last_run_ts}'"
                )

        loaded_documents = []
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(final_query)
                    rows = cur.fetchall()
                    if not rows:
                        logger.info("No new data found in the database.")
                        return []

                    for row in rows:
                        row_dict = dict(row)
                        content_key = list(row_dict.keys())[0]
                        content = row_dict.pop(content_key)
                        metadata = row_dict
                        metadata["source"] = (
                            f"postgres://{self.db_params['user']}@{self.db_params['host']}/{self.db_params['database']}"
                        )
                        doc = Document(content=content, metadata=metadata)
                        loaded_documents.append(doc)
            return loaded_documents
        except psycopg2.Error as e:
            logger.error(
                f"Error loading data from PostgreSQL: {e}", exc_info=True
            )
            return []

    def update_state(self, processed_docs: List[Document]):
        self.state_manager.update_run_timestamp()

    def test_connection(self):
        logger.info("Testing connection to PostgreSQL database")
        try:
            with psycopg2.connect(**self.db_params) as conn:
                logger.info("Connection to PostgreSQL successful")
        except psycopg2.Error as e:
            raise ConnectionError("Failed to connect to PostgreSQL") from e
