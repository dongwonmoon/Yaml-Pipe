from abc import ABC, abstractmethod
import pandas as pd
import lancedb
from lancedb.pydantic import pydantic_to_schema, Vector
from pydantic import BaseModel


class BaseSink(ABC):
    @abstractmethod
    def sink(self, data: pd.DataFrame):
        pass


class LanceDBSink(BaseSink):
    def __init__(self, uri: str, table_name: str):
        self.uri = uri
        self.table_name = table_name

    def sink(self, data: pd.DataFrame):
        """
        입력받은 DataFrame을 LanceDB 테이블에 저장합니다.
        """
        print(
            f"LanceDB에 데이터를 저장합니다. (URI: {self.uri}, 테이블: {self.table_name})"
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
        print("벡터 DB 저장 완료")
