from abc import ABC, abstractmethod
import pandas as pd
import lancedb
import logging
from lancedb.pydantic import pydantic_to_schema, Vector
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BaseSink(ABC):
    @abstractmethod
    def sink(self, data: pd.DataFrame):
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
