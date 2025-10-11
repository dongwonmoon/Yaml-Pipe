import json
import logging
from typing import List
import pandas as pd
import lancedb
import chromadb

from ..components.embedders import BaseEmbedder
from .data_models import Document

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, embedder: BaseEmbedder, sink_config: dict):
        self.embedder = embedder
        self.sink_config = sink_config
        self.sink_type = self.sink_config.get("type")

        if self.sink_type == "lancedb":
            db = lancedb.connect(self.sink_config["config"]["uri"])
            self.retriever = db.open_table(self.sink_config["config"]["table_name"])
        elif self.sink_type == "chromadb":
            client = chromadb.PersistentClient(path=self.sink_config["config"]["path"])
            self.retriever = client.get_collection(
                self.sink_config["config"]["collection_name"]
            )
        else:
            raise ValueError(f"Unsupported sink type: {self.sink_type}")

    def _search(self, query: str, k: int) -> List[dict]:
        query_vector = self.embedder.embed([query])[0]

        if self.sink_type == "lancedb":
            results = self.retriever.search(query_vector).limit(k).to_df()
            return results.to_dict("records")
        elif self.sink_type == "chromadb":
            results = self.retriever.query(
                query_embeddings=[query_vector], n_results=k
            ).tolist()
            print(results)
            return results["metadatas"][0]

    def evaluate(self, dataset_path: str, k: int = 5) -> dict:
        logger.info(f"'{dataset_path}' evaluation started ...")

        with open(dataset_path, "r") as f:
            eval_data = [json.loads(line) for line in f]

        hit_count = 0
        for item in eval_data:
            question = item["question"]
            expected_source = item["expected_source"]

            search_results = self._search(question, k)

            for result in search_results:
                if result["source"] == expected_source:
                    hit_count += 1
                    break

        hit_rate = (hit_count / len(eval_data)) * 100 if eval_data else 0

        logger.info(
            f"Evaluation Finished. Hit Rate: {hit_rate:.2f}% ({hit_count}/{len(eval_data)})"
        )
        return {
            "hit_rate": hit_rate,
            "total_questions": len(eval_data),
            "hits": hit_count,
        }
