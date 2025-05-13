import numpy as np
from tokenizer import Tokenizer
from vocabulary import Vocabulary
from indexer import Indexer
import heapq

import json
from util import (
    get_memory_usage,
    IDF,
    parse_args,
    PLN,
    TF as TF,
)

import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count, listdir, makedirs, path

n_threads = 10

json_format = {
    "Query": "",
    "Results": [
        {"ID": 0, "Score": 0.0},
    ],
}


class Processor:
    """
    Class to handle processing of queries against an index.
    """

    def __init__(
        self,
        ranker,
        index,
        query="queries.txt",
        memory=1024,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        """
        Initialize the Processor with the given parameters.

        Parameters
        ----------
        ranker: str
            The ranking function to be used for scoring documents. Must be either 'TFIDF' or 'BM25'.
        query: str
            The path to the file containing the queries to be processed.
        """
        self.query_file = query
        self.memory = memory
        self.index_path = index
        self.queries = []
        self.vocabulary = Vocabulary()
        self.vocabulary.read_vocabulary()
        self.k1 = k1
        self.b = b
        if ranker.lower() == "tfidf":
            self.ranker = self.tfidf
        elif ranker.lower() == "bm25":
            self.ranker = self.bm25
        else:
            raise ValueError("Ranker must be 'TFIDF' or 'BM25'")

        with open("info.json", "r") as f:
            self.docs_info = json.load(f, object_hook=dict)
            self.average_doc_len = np.mean(list(self.docs_info.values()))

    def read_query(self):
        """
        Read the queries from the query file and store them in a list.
        """
        with open(self.query_file, "r") as f:
            tokenizer = Tokenizer()
            for line in f:
                self.queries.append(tokenizer.tokenize_text(line))

    def print_query(self, i=0):
        """
        Print the queries to the console.
        """
        if i >= len(self.queries):
            print("No more queries to print.")
            return
        query = self.queries[i]
        print(f"Query {i}: {query}")

    def get_line(self, word: str):
        """
        Get the line from the inverted index for a given word.

        Parameters
        ----------
        word: str
            The word for which to get the line from the inverted index.

        Returns
        -------
        dict
            The line from the inverted index for the given word.
        """
        return Indexer().get_line(word)

    def tfidf(self, query: list[str], doc_id: str, indexes: dict) -> float:
        """
        Calculate the TF-IDF score for a given query.

        Parameters
        ----------
        query: list[str]
            List of words from the query.
        doc_id: str
            The ID of the document in which to calculate the TF-IDF score.
        index: dict
            The indexes of the words of the query.

        Returns
        -------
        float
            The TF-IDF score for the query for the document.
        """
        tf_idf = 0.0
        for word in query:
            index = indexes[word]
            if index:
                if doc_id not in index:
                    word_count = 0
                else:
                    word_count = index[doc_id]

                tf = TF(word_count)
                idf = IDF(len(self.docs_info), len(index))
                tf_idf += tf * idf
            else:
                print(f"Word '{word}' not found.")

        return float(tf_idf)

    def bm25(self, query: list[str], doc_id: str, indexes: dict) -> float:
        """
        Calculate the BM25 score for a given query.

        Parameters
        ----------
        query: list[str]
            List of words from the query.
        doc_id: str
            The ID of the document in which to calculate the BM25 score.
        index: dict
            The indexes of the words of the query.

        Returns
        -------
        float
            The BM25 score for the query for the document.
        """
        bm_25 = 0.0
        doc_len = self.docs_info[doc_id]
        for word in query:
            index = indexes[word]
            if index:
                if doc_id not in index:
                    word_count = 0
                else:
                    word_count = index[doc_id]

                idf = IDF(len(self.docs_info), len(index))
                pln = PLN(self.b, doc_len, self.average_doc_len)
                bm_25 += (
                    ((self.k1 + 1) * word_count) / (word_count + self.k1 * pln)
                ) * idf
            else:
                print(f"Word '{word}' not found.")
        return float(bm_25)

    def daat(self, i: int = 0, top_k: int = 10):
        """
        Calculate the scores of the documents contaning the words present in the query.
        Using Document-at-a-time.

        Parameters
        ----------
        i: int
            The current query.
        top_k: int
            The number of top results.

        Returns
        -------
        list
            The top k results with their doc_ids and scores.
        """
        results = []
        lists = [self.get_line(word) for word in self.queries[i]]
        indexes = dict(zip(self.queries[i], lists))
        targets = {doc_id for word in self.queries[i] for doc_id in indexes[word]}

        for target in targets:
            score = 0
            for postings in lists:
                for doc_id in postings.keys():
                    if doc_id == target:
                        score += self.ranker(self.queries[i], doc_id, indexes)

            if len(results) < top_k:
                heapq.heappush(results, (score, target))
            
            elif score > min_score:
                heapq.heappop(results)
                heapq.heappush(results, (score, target))

            min_score = results[0][0] if results else 0

        for result in sorted(results, reverse=True):
            print(f"DocId: {result[1]} - Score: {result[0]}")

        return results


if __name__ == "__main__":
    mem = [get_memory_usage()]
    args = parse_args("processor")
    processor = Processor(args.Ranker, args.Index, args.Query)
    processor.read_query()
    processor.print_query()
    processor.daat()
