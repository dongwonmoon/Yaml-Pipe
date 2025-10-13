import json
import logging
from typing import List, Dict, Any

import pandas as pd
import lancedb
import chromadb

from ..components.embedders import BaseEmbedder
from ..utils.data_models import Document

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates the performance of a retriever against a given dataset.
    """

    def __init__(self, embedder: BaseEmbedder, sink_config: dict):
        """
        Initializes the Evaluator.

        Args:
            embedder: The embedder to use for generating query vectors.
            sink_config: The configuration for the sink (vector database).
        """
        self.embedder = embedder
        self.sink_config = sink_config
        self.sink_type = self.sink_config.get("type")
        self.retriever = self._init_retriever()

    def _init_retriever(self):
        """Initializes the retriever based on the sink configuration."""
        if self.sink_type == "lancedb":
            db = lancedb.connect(self.sink_config["config"]["uri"])
            return db.open_table(self.sink_config["config"]["table_name"])
        elif self.sink_type == "chromadb":
            client = chromadb.PersistentClient(
                path=self.sink_config["config"]["path"]
            )
            return client.get_collection(
                self.sink_config["config"]["collection_name"]
            )
        else:
            raise ValueError(f"Unsupported sink type: {self.sink_type}")

    def _search(self, query: str, k: int) -> List[dict]:
        """
        Performs a search on the retriever.

        Args:
            query: The query string.
            k: The number of results to retrieve.

        Returns:
            A list of search results.
        """
        query_vector = self.embedder.embed([query])[0]

        if self.sink_type == "lancedb":
            results = self.retriever.search(query_vector).limit(k).to_df()
            return results.to_dict("records")
        elif self.sink_type == "chromadb":
            results = self.retriever.query(
                query_embeddings=[query_vector.tolist()], n_results=k
            )
            return results["metadatas"][0]

    def evaluate(self, dataset_path: str, k: int = 5) -> Dict[str, Any]:
        """
        Evaluates the retriever on a given dataset.

        Args:
            dataset_path: The path to the evaluation dataset.
            k: The number of results to retrieve for each query.

        Returns:
            A dictionary containing the evaluation results (hit_rate, total_questions, hits).
        """
        logger.info(f"Starting evaluation for dataset: '{dataset_path}'")

        with open(dataset_path, "r") as f:
            eval_data = [json.loads(line) for line in f]

        hit_count = 0
        for item in eval_data:
            question = item["question"]
            expected_source = item["expected_source"]

            search_results = self._search(question, k)

            for result in search_results:
                if result["source"] == expected_source:
                    logger.debug(
                        f"Found expected source '{expected_source}' for question '{question}'"
                    )
                    hit_count += 1
                    break

        if not eval_data:
            hit_rate = 0.0
        else:
            hit_rate = (hit_count / len(eval_data)) * 100

        logger.info(
            f"Evaluation Finished. Hit Rate: {hit_rate:.2f}% ({hit_count}/{len(eval_data)})"
        )
        return {
            "hit_rate": hit_rate,
            "total_questions": len(eval_data),
            "hits": hit_count,
        }
