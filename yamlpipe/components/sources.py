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

        This method should handle fetching data, parsing it, and wrapping it
        in Document objects. It should also interact with a StateManager to
        avoid reprocessing unchanged data.

        Returns:
            List[Document]: A list of Document objects, each representing a loaded item.
        """
        pass

    @abstractmethod
    def test_connection(self):
        """
        Tests the connection to the data source to ensure it is accessible.

        Raises:
            Exception: If the connection test fails.
        """
        pass


class LocalFileSource(BaseSource):
    """
    Loads documents from the local filesystem.

    This source scans a directory for files matching a glob pattern and uses the
    `unstructured` library to parse content. It tracks file modifications
    using a StateManager to process only new or changed files.
    """

    def __init__(self, path: str, glob_pattern: str, state_manager: StateManager):
        """
        Initializes the LocalFileSource.

        Args:
            path (str): The directory path to search for files.
            glob_pattern (str): The glob pattern to match files within the directory.
            state_manager (StateManager): The state manager to track file changes.
        """
        self.path = Path(path)
        self.glob_pattern = glob_pattern
        self.state_manager = state_manager
        logger.debug(
            f"Initialized LocalFileSource with path='{self.path}' and glob='{self.glob_pattern}'"
        )

    def load_data(self) -> List[Document]:
        """
        Loads all files matching the glob pattern from the specified path,
        filters out files that have not changed since the last run, and
        returns a list of Document objects for the new or modified files.
        """
        logger.info(
            f"Scanning for files in '{self.path}' with pattern '{self.glob_pattern}'."
        )
        if not self.path.is_dir():
            logger.error(
                f"Source path '{self.path}' is not a valid directory. Aborting."
            )
            return []

        all_files = [str(f) for f in self.path.glob(self.glob_pattern) if f.is_file()]
        logger.debug(f"Found {len(all_files)} total files matching glob pattern.")

        new_or_changed_files = [
            f for f in all_files if self.state_manager.has_changed(f)
        ]

        if not new_or_changed_files:
            logger.info("No new or changed files detected.")
            return []

        logger.info(
            f"Found {len(new_or_changed_files)} new or changed files to process."
        )

        loaded_data = []
        for file_path_str in new_or_changed_files:
            try:
                logger.debug(f"Partitioning file: {file_path_str}")
                elements = partition(filename=file_path_str)
                content = "\n\n".join([str(el) for el in elements])

                if not content.strip():
                    logger.warning(
                        f"File '{file_path_str}' is empty or contains no text. Skipping."
                    )
                    continue

                doc = Document(content=content, metadata={"source": file_path_str})
                loaded_data.append(doc)
                logger.debug(
                    f"Successfully loaded and partitioned file: {file_path_str}"
                )

            except Exception as e:
                logger.error(
                    f"Error processing file '{file_path_str}': {e}",
                    exc_info=True,
                )

        logger.info(f"Successfully loaded {len(loaded_data)} documents.")
        return loaded_data

    def test_connection(self):
        """Tests if the source directory exists and is accessible."""
        logger.info(f"Testing connection for LocalFileSource at path: {self.path}")
        if not self.path.exists():
            raise FileNotFoundError(f"Source path '{self.path}' does not exist.")
        if not self.path.is_dir():
            raise NotADirectoryError(f"Source path '{self.path}' is not a directory.")
        logger.info("Connection to LocalFileSource successful.")


class WebSource(BaseSource):
    """
    Loads a document from a web URL.

    This source fetches the content of a single web page, parses the HTML to
    extract clean text, and returns it as one Document.
    """

    def __init__(self, url: str):
        """
        Initializes the WebSource.

        Args:
            url (str): The URL of the web page to load.
        """
        self.url = url
        logger.debug(f"Initialized WebSource with URL: {self.url}")

    def load_data(self) -> List[Document]:
        """
        Fetches content from the URL, parses it, and returns a single Document.
        """
        logger.info(f"Fetching content from URL: {self.url}")
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())
            clean_text = "\n".join(line for line in lines if line)

            if not clean_text.strip():
                logger.warning(f"No text content found at URL: {self.url}")
                return []

            doc = Document(content=clean_text, metadata={"source": self.url})
            logger.info(f"Successfully fetched and parsed content from: {self.url}")
            return [doc]

        except requests.exceptions.RequestException as e:
            logger.error(
                f"Failed to fetch content from URL '{self.url}': {e}",
                exc_info=True,
            )
            return []

    def test_connection(self):
        """Tests if the web URL is reachable by sending a HEAD request."""
        logger.info(f"Testing connection for WebSource at URL: {self.url}")
        try:
            response = requests.head(self.url, timeout=5)
            response.raise_for_status()
            logger.info("Connection to WebSource successful.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to URL '{self.url}': {e}", exc_info=True)
            raise ConnectionError(f"Failed to connect to URL: {self.url}")


class S3Source(BaseSource):
    """
    Loads documents from an AWS S3 bucket.

    This source lists objects under a specified prefix in an S3 bucket. It uses the
    object's ETag hash, managed by a StateManager, to process only new or
    changed objects, similar to how LocalFileSource tracks local file hashes.
    """

    def __init__(self, bucket: str, prefix: str, state_manager: StateManager):
        """
        Initializes the S3Source.

        Args:
            bucket (str): The name of the S3 bucket.
            prefix (str): The prefix (folder path) within the bucket to scan for objects.
            state_manager (StateManager): The state manager to track object changes.
        """
        self.bucket_name = bucket
        self.prefix = prefix
        self.state_manager = state_manager
        self.s3_client = boto3.client("s3")

    def load_data(self) -> List[Document]:
        """
        Loads all objects from the S3 bucket/prefix that have changed since the last run.

        It compares the ETag of each object with the stored ETag in the state manager
        to determine if the object is new or has been modified.
        """
        logger.info(f"Loading data from S3 bucket: {self.bucket_name}")

        try:
            # List all objects within the specified bucket and prefix.
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=self.prefix
            )
            all_objects = response.get("Contents", [])
        except ClientError as e:
            logger.error(f"Error listing objects in S3 bucket: {e}", exc_info=True)
            return []

        # Identify objects that are new or have a different ETag than the last run.
        new_or_changed_objects = []
        for obj in all_objects:
            obj_key = obj["Key"]
            # ETag is an identifier for a specific version of an object.
            obj_etag = obj["ETag"].strip("'")

            # The source identifier for S3 objects is their full s3:// path.
            source_id = f"s3://{self.bucket_name}/{obj_key}"
            last_etag = self.state_manager.state["processed_files"].get(source_id)

            if obj_etag != last_etag:
                new_or_changed_objects.append(obj)

        if not new_or_changed_objects:
            logger.info("No new or changed objects detected in S3.")
            return []

        logger.info(
            f"Found {len(new_or_changed_objects)} new or changed objects to process."
        )

        # Download and read the content of new/changed objects.
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
                    metadata={"source": f"s3://{self.bucket_name}/{obj_key}"},
                )
                loaded_documents.append(doc)

            except Exception as e:
                logger.error(f"Error loading object {obj_key}: {e}", exc_info=True)

        return loaded_documents

    def test_connection(self):
        """
        Tests the connection to S3 by checking if the bucket is accessible.
        Also verifies that AWS credentials are configured.
        """
        logger.info(f"Testing connection to S3 bucket: {self.bucket_name}")
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info("Connection to S3 bucket successful.")
        except NoCredentialsError:
            raise Exception("AWS credentials not found. Please configure boto3.")
        except ClientError as e:
            # If a ClientError is caught, it can mean the bucket does not exist or is forbidden.
            logger.error(f"Error testing connection to S3 bucket: {e}", exc_info=True)
            raise ConnectionError(f"Failed to connect to S3 bucket: {self.bucket_name}")


class PostgreSQLSource(BaseSource):
    """
    Loads data from a PostgreSQL database using a specified SQL query.

    Each row returned by the query is treated as a separate document. The first
    column of the query result is used as the main content, and the remaining
    columns are added to the document's metadata.
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
        """
        Initializes the PostgreSQLSource with database connection details and a query.

        Args:
            host (str): The database host.
            port (int): The database port.
            database (str): The name of the database.
            user (str): The username for authentication.
            password (str): The password for authentication.
            query (str): The SQL query to execute to fetch the data.
        """
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
        """
        Connects to the database, executes the configured query, and returns the
        results as a list of Document objects.
        """
        logger.info("Loading data from PostgreSQL database")

        last_run_ts = self.state_manager.get_last_run_timestamp()

        final_query = self.query
        if last_run_ts:
            if "where" in self.query.lower():
                final_query += f" AND {self.timestamp_column} > '{last_run_ts}'"
            else:
                final_query += f" WHERE {self.timestamp_column} > '{last_run_ts}'"

        loaded_documents = []
        conn = None

        try:
            # Establish the database connection.
            conn = psycopg2.connect(**self.db_params)
            # Use DictCursor to get rows as dictionaries (column_name: value).
            cur = conn.cursor(cursor_factory=DictCursor)
            cur.execute(self.query)
            rows = cur.fetchall()

            if not rows:
                logger.info("No data found in the database for the given query.")
                return []

            # Process each row from the query result.
            for i, row in enumerate(rows):
                row_dict = dict(row)

                # The first column is assumed to be the main content.
                content_key = list(row_dict.keys())[0]
                content = row_dict.pop(content_key)

                # The rest of the columns are treated as metadata.
                metadata = row_dict
                metadata["source"] = (
                    f"postgres://{self.db_params['user']}@{self.db_params['host']}/{self.db_params['database']}"
                )

                doc = Document(content=content, metadata=metadata)
                loaded_documents.append(doc)

            cur.close()
            return loaded_documents

        except psycopg2.Error as e:
            logger.error(f"Error loading data from PostgreSQL: {e}", exc_info=True)
            return []

        finally:
            # Ensure the connection is always closed.
            if conn:
                conn.close()

    def test_connection(self):
        """
        Tests the connection to the PostgreSQL database by attempting to connect.
        """
        logger.info("Testing connection to PostgreSQL database")
        conn = None
        try:
            conn = psycopg2.connect(**self.db_params)
            conn.close()
            logger.info("Connection to PostgreSQL successful")
        except psycopg2.Error as e:
            logger.error(f"Error testing connection to PostgreSQL: {e}", exc_info=True)
            raise ConnectionError("Failed to connect to PostgreSQL")
        finally:
            if conn:
                conn.close()
