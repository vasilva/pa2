from tokenizer import Tokenizer
from vocabulary import Vocabulary

import json
import heapq
from util import *

import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import listdir, makedirs, path

N_THREADS = 32
MAX_DOCS = 4_641_784


class Indexer:
    """
    Class to handle indexing of a directory of files.
    """

    def __init__(
        self,
        memory: int = 1024,
        corpus_path: str = "corpus.jsonl",
        index_path: str = "index",
        batch_size: int = 100,
        max_doc: int = 10000,
    ):
        """
        Initialize the Indexer with the given parameters.

        Parameters
        ----------
        memory: int
            The maximum memory to be used for the indexer.
        corpus_path: str
            The path to the corpus file.
        index_path: str
            The path to the directory where the index files will be stored.
        batch_size: int
            The number of documents to be processed in each batch.
        max_doc: int
            The maximum number of documents to be processed. If -1, process all documents.
        """
        self.memory = memory * 1024 * 1024
        self.corpus = corpus_path
        self.index_path = index_path
        self.tokenizer = Tokenizer()
        self.vocabulary = Vocabulary()
        self.batch_size = batch_size
        self.max_doc = max_doc
        self.n_docs = 0
        self.docs_len = dict()
        self.timer = 0
        self.n_lists = 0
        self.n_postings = 0
        self.memory_usage = []

    def parse_documents(self):
        """
        Parse the corpus file and create an inverted index.
        """
        self.timer = time.monotonic()
        heapq.heappush(self.memory_usage, get_memory_usage())
        with open(self.corpus, "r", encoding="utf8") as f:
            data = [[] for _ in range(N_THREADS)]
            # Read the file line by line
            for i, line in enumerate(f):
                if self.max_doc < 0:
                    # If max_doc is negative, process all documents
                    pass
                elif i >= self.max_doc:
                    # If max_doc is reached, stop processing
                    break

                # Distribute the file in batches
                doc = json.loads(line)
                self.n_docs += 1
                doc_len = len(
                    str(
                        doc["title"]
                        + " "
                        + doc["text"]
                        + " ".join(doc["keywords"])
                        + " "
                        + " ".join(doc["keywords"])
                    ).split()
                )
                self.docs_len[doc["id"]] = doc_len
                data[i % len(data)].append(doc)

                if len(data[-1]) >= self.batch_size or i == self.max_doc:
                    # Process batch
                    max_heap = [-x for x in self.memory_usage]
                    max_mem = heapq.heappop(max_heap)
                    if -max_mem > self.memory:
                        raise MemoryError(
                            f"Out of memory: {max_mem // (1024 * 1024)} MB used."
                        )

                    self.n_lists += 1
                    index = self.process_batch(data)
                    index_part = dict(sorted(self.merge_index(index).items()))
                    data = [[] for _ in range(N_THREADS)]
                    self.write_index(index_part, self.n_lists - 1)
                    self.n_postings += self.get_postings(index_part)
                    heapq.heappush(self.memory_usage, get_memory_usage())

        with open("info.json", "w") as f:
            json.dump(self.docs_len, f)

        self.timer = time.monotonic() - self.timer

    def process_batch(self, data):
        """
        Process the documents in the corpus and create an inverted index.

        Parameters
        ----------
        data: list
            A list of documents to be processed.

        Returns
        -------
        list[dict]
            A list of inverted indexes for each document.
        """
        index_results, vocab_results = [], []

        # Process the documents in parallel
        with ThreadPoolExecutor(max_workers=len(data)) as executor:
            futures = [executor.submit(self.process_documents, batch) for batch in data]

            # Wait for all futures to complete
            for future in as_completed(futures):
                try:
                    partial_index, vocabulary = future.result()
                    index_results.append(partial_index)
                    vocab_results.append(vocabulary)

                except Exception(f"Thread {future} failed to produce result"):
                    continue

            for vocab in vocab_results:
                self.vocabulary.merge(vocab)

        return index_results

    def process_documents(self, docs_list: list):
        """
        Process a list of documents and create an inverted index and the vocabulary.

        Parameters
        ----------
        docs_list : list
            A list of documents to be processed.

        Returns
        -------
        tuple[dict, Vocabulary]
            A tuple containing the inverted index and the vocabulary.
        """
        indexes, vocabularies = [], []
        for doc in docs_list:
            # Create a partial inverted index for the document
            partial_inverted_index, vocabulary = self.make_partial_index(doc)
            indexes.append(partial_inverted_index)
            vocabularies.append(vocabulary)

        # Merge the partial inverted indexes and vocabularies
        vocabulary = Vocabulary()
        for v in vocabularies:
            vocabulary.merge(v)

        return self.merge_index(indexes), vocabulary

    def make_partial_index(self, doc: dict):
        """
        Add the document to the inverted index and its words to the vocabulary.

        Parameters
        ----------
        doc: dict
            The document to be added to the inverted index.

        Returns
        -------
        tuple[dict, Vocabulary]
            A tuple containing the inverted index and the vocabulary.
        """
        # Tokenize the title, text, and keywords
        doc_id = int(doc["id"])
        document = str(
            doc["title"]
            + " "
            + doc["text"]
            + " "
            + " ".join(doc["keywords"])
            + " "
            + " ".join(doc["keywords"])
        )
        tokens = self.tokenizer.tokenize_text(document)
        inverted_index = dict()
        # Add the tokens to the vocabulary
        vocabulary = Vocabulary()
        vocabulary.add(tokens)

        # Add the tokens to the inverted index
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = {doc_id: 1}

            else:
                inverted_index[token][doc_id] = inverted_index[token].get(doc_id, 0) + 1

        return inverted_index, vocabulary

    def merge_index(self, indexes_list: list):
        """
        Merge the list of partial inverted indexes into one inverted index.

        Parameters
        ----------
        indexes_list: list
            A list of partial inverted indexes to be merged.

        Returns
        -------
        dict
            The merged inverted index.
        """
        merged_index = dict()
        for index in indexes_list:
            for token, doc_ids in index.items():
                if token not in merged_index:
                    merged_index[token] = doc_ids

                else:
                    for doc_id, freq in doc_ids.items():
                        merged_index[token][doc_id] = (
                            merged_index[token].get(doc_id, 0) + freq
                        )

        return merged_index

    def get_line(self, word: str):
        """
        Get the line of the inverted index for a given word.

        Parameters
        ----------
        word: str
            The word to get the line for.

        Returns
        -------
        dict
            The line of the inverted index for the word.
        """
        if not self.vocabulary:
            # If the vocabulary is empty, read it from the file
            self.vocabulary.read_vocabulary()

        if word not in self.vocabulary:
            return None

        word_index = []
        parts = listdir(self.index_path)
        for i in range(len(parts)):
            partial_index = self.read_index(i)
            if word in partial_index:
                word_index.append(partial_index[word])

        sorted_word_index = dict(
            sorted(
                self.merge_index(word_index).items(),
                key=lambda x: int(x[0]),
            )
        )
        return sorted_word_index

    def write_index(self, inverted_index: dict, part: int = 0):
        """
        Write the inverted index to a file.

        Parameters
        ----------
        inverted_index: dict
            The inverted index to be written to a file.
        part: int
            The part of the inverted index to write.
        """
        if not path.exists(self.index_path):
            makedirs(self.index_path)

        file_name = f"{self.index_path}/inverted_index_{part}.json"
        with open(file_name, "w") as f:
            json.dump(inverted_index, f, indent=2)

    def read_index(self, part: int = 0):
        """
        Read the inverted index from a file.

        Parameters
        ----------
        part: int
            The part of the inverted index to read.

        Returns
        -------
        dict
            The inverted index.
        """
        file_name = f"{self.index_path}/inverted_index_{part}.json"
        with open(file_name, "r") as f:
            return json.load(f)

    def get_postings(self, index: dict):
        n_postings = 0
        for word_list in index:
            n_postings += len(word_list)

        return n_postings / len(index)

    def print_results(self):
        """
        Print a JSON document to the standard output.
        """
        index = get_files_size(self.index_path)
        avls = self.n_postings / self.n_lists
        print('{ "Index Size": %d' % index)
        print('  "Elapsed Time": %d,' % self.timer)
        print('  "Number of Lists": %d,' % self.n_lists)
        print('  "Average List Size": %.2f }' % avls)


if __name__ == "__main__":
    args = parse_args("indexer")
    max_memory = args.Memory * 1024 * 1024

    batch_size = args.Memory // 4
    indexer = Indexer(max_memory, args.Corpus, args.Index, batch_size, MAX_DOCS)
    tokenizer = indexer.tokenizer
    vocabulary = indexer.vocabulary

    indexer.parse_documents()
    vocabulary.write_vocabulary()
    memory_usage = indexer.memory_usage

    indexer.print_results()

    with open("mem.txt", "w") as m:
        for mem in memory_usage:
            m.write(f"{mem // (1024**2)} MB\n")
