import numpy as np
from tokenizer import Tokenizer
from vocabulary import Vocabulary
from indexer import Indexer
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from util import *


class Processor:
    """
    Class to handle processing of queries against an index.
    """

    def __init__(
        self,
        ranker: str,
        index_path: str,
        query: str = "queries.txt",
        k1: float = 1.2,
        b: float = 0.75,
    ):
        """
        Initialize the Processor with the given parameters.

        Parameters
        ----------
        ranker: str
            The ranking function to be used for scoring documents. It can be either `'TFIDF'`, `'BM25'` or `'PLNVSM'`.
        query: str
            The path to the file containing the queries to be processed.
        k1: float
            Hyper-parameter for the BM25 score.
        b: float
            Hyper-parameter for the Pivoted Length Normalization.
        """
        self.query_file = query
        self.index_path = index_path
        self.vocabulary = Vocabulary()
        self.vocabulary.read_vocabulary()
        self.k1 = k1
        self.b = b
        self.results = []

        # Ranker selection
        if ranker.lower() == "tfidf":
            self.ranker = self._tfidf
        elif ranker.lower() == "bm25":
            self.ranker = self._bm25
        elif ranker.lower() == "plnvsm":
            self.ranker = self._plnvsm
        else:
            raise ValueError("Ranker must be 'TFIDF', 'BM25' or 'PLNVSM'")

        # Info for documents size and average document length
        with open("data/info.json", "r") as f:
            self.docs_info = json.load(f, object_hook=dict)
            self.average_doc_len = np.mean(list(self.docs_info.values()))

        self.tokenized_queries = []
        with open(self.query_file, "r") as f:
            self.queries = f.readlines()
            tokenizer = Tokenizer()
            for query in self.queries:
                self.tokenized_queries.append(tokenizer.tokenize_text(query))

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

    def _tfidf(self, query: list[str], doc_id: str, indexes: dict) -> float:
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
                if doc_id in index:
                    word_count = index[doc_id]
                else:
                    word_count = 0

                tf = TF_1(word_count)
                idf = IDF(len(self.docs_info), len(index))
                tf_idf += tf * idf

        return float(tf_idf)

    def _plnvsm(self, query: list[str], doc_id: str, indexes: dict) -> float:
        """
        Calculate the Pivoted Length Normalization VSM score for a given query.

        Parameters
        ----------
        query: list[str]
            List of words from the query.
        doc_id: str
            The ID of the document in which to calculate the PLN VSM score.
        index: dict
            The indexes of the words of the query.

        Returns
        -------
        float
            The PLN VSM score for the query for the document.
        """
        plm_vsm = 0.0
        doc_len = self.docs_info[doc_id.zfill(7)]
        for word in query:
            index = indexes[word]
            if index:
                if doc_id in index:
                    word_count = index[doc_id]
                else:
                    word_count = 0

            tf = TF_2(word_count)
            idf = IDF(len(self.docs_info), len(index))
            pln = PLN(self.b, doc_len, self.average_doc_len)
            plm_vsm += tf * idf / pln

        return float(plm_vsm)

    def _bm25(self, query: list[str], doc_id: str, indexes: dict) -> float:
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
        doc_len = self.docs_info[doc_id.zfill(7)]
        for word in query:
            index = indexes[word]
            if index:
                if doc_id in index:
                    word_count = index[doc_id]
                else:
                    word_count = 0

                idf = IDF(len(self.docs_info), len(index))
                pln = PLN(self.b, doc_len, self.average_doc_len)
                bm_25 += (self.k1 + 1) * word_count * idf / (word_count + self.k1 * pln)

        return float(bm_25)

    def _score(self, target: str, indexes: dict, query: list) -> float:
        """
        Calculate the score of the target document.

        Parameters
        ----------
        target: str
            The target document.
        indexes: dict
            Word/Index Dictionary
        query: list
            The current query

        Returns
        -------
        float
            The score using either TF-IDF, BM25 or PLN VSM rankers.
        """
        lists = indexes.values()
        score = 0.0
        for postings in lists:
            for doc_id in postings.keys():
                if doc_id == target:
                    score += self.ranker(query, doc_id, indexes)

        return score

    def daat(self, current_query: int = 0, top_k: int = 10):
        """
        Calculate the scores of the documents contaning the words present in the query.
        Using Document-at-a-time.

        Parameters
        ----------
        current_query: int
            The current query.
        top_k: int
            The number of top results.

        Returns
        -------
        list
            The top k results with their doc_ids and scores.
        """
        results, lists = [], []
        # One thread for each word of the query.
        with ThreadPoolExecutor(
            max_workers=len(self.queries[current_query])
        ) as executor:
            futures = [
                executor.submit(self.get_line, word)
                for word in self.tokenized_queries[current_query]
            ]
            for future in as_completed(futures):
                try:
                    index = future.result()
                    lists.append(index)

                except Exception:
                    continue

            indexes = dict(zip(self.tokenized_queries[current_query], lists))
            targets = []
            for index in indexes.values():
                targets.append(set(index.keys()))

            # Filter to targets that contain all terms of the query
            convunctive_targets = set()
            for j in range(len(targets)):
                convunctive_targets = targets[0].intersection(targets[j])

            lists.clear()
            targets.clear()

            for target in convunctive_targets:
                score = self._score(
                    target, indexes, self.tokenized_queries[current_query]
                )

                # Keep the heap with k elements max
                if len(results) < top_k:
                    heapq.heappush(results, (score, target))

                elif score > min_score:
                    heapq.heapreplace(results, (score, target))

                min_score = results[0][0] if results else 0.0

        self.results.append(sorted(results, reverse=True))

    def print_results(self, current_query: int):
        """
        Print a JSON document to standard output with the top
        results retrieved for that query according to the following format:
        * `Query`: The query text;
        * `Results`: A list of results.
        
        Each result in the Results list must be represented with the fields:
        * `ID`: The respective result ID;
        * `Score`: The final document score.
        """
        results = self.results[current_query]
        query = self.queries[current_query]

        print('{ "Query": "%s",' % query)
        print('  "Results": [')
        for result in results:
            print('    { "ID": "%s",' % result[1].zfill(7))
            print('      "Score": %.2f },' % result[0])

        print("  ]")
        print("}")


if __name__ == "__main__":
    mem = [get_memory_usage()]
    args = parse_args("processor")
    processor = Processor(args.Ranker, args.Index, args.Query)

    for i in range(len(processor.queries)):
        processor.daat(i)
        processor.print_results(i)
