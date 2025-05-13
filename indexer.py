from tokenizer import Tokenizer
from vocabulary import Vocabulary

import json
from util import get_memory_usage, parse_args, TF as TF

import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count, listdir, makedirs, path

n_threads = 10

json_format = {
    "Index Size": 0,
    "Elapsed Time": 0,
    "Number of Lists": 0,
    "Average List Size": 0.0,
}


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
        self.memory = memory
        self.corpus = corpus_path
        self.index_path = index_path
        self.tokenizer = Tokenizer()
        self.vocabulary = Vocabulary()
        self.batch_size = batch_size
        self.max_doc = max_doc
        self.n_docs = 0
        self.docs_len = dict()

    def parse_documents(self):
        """
        Parse the corpus file and create an inverted index.
        """
        with open(self.corpus, "r", encoding="utf8") as f:
            data = [[] for _ in range(n_threads)]
            j = 0
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
                        doc["title"] + " " + doc["text"] + " ".join(doc["keywords"])
                    ).split()
                )
                self.docs_len[int(doc["id"])] = doc_len
                data[i % len(data)].append(doc)

                if len(data[-1]) >= self.batch_size or i == self.max_doc:
                    print(f"Batch {j}, mem: {get_memory_usage()} B")
                    index = self.process_batch(data)
                    index_part = dict(sorted(self.merge_inverted_index(index).items()))
                    data = [[] for _ in range(n_threads)]
                    self.write_index(index_part, j)
                    j += 1

        with open("info.json", "w") as f:
            json.dump(self.docs_len, f)

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

                except Exception as e:
                    print(f"Error processing document: {e}")
                    continue

            for vocab in vocab_results:
                self.vocabulary.merge(vocab)

        return index_results

    def process_documents(self, docs_list):
        """
        Process a list of documents and create an inverted index.

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
            partial_inverted_index, vocabulary = self.make_partial_inverted_index(doc)
            indexes.append(partial_inverted_index)
            vocabularies.append(vocabulary)

        # Merge the partial inverted indexes and vocabularies
        vocabulary = Vocabulary()
        for v in vocabularies:
            vocabulary.merge(v)

        return self.merge_inverted_index(indexes), vocabulary

    def make_partial_inverted_index(self, doc):
        """
        Add the document to the inverted index.

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
            doc["title"] + " " + doc["text"] + " " + " ".join(doc["keywords"])
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

    def merge_inverted_index(self, indexes_list):
        """
        Merge the partial inverted indexes into one inverted index.

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

    def __str__(self):
        """
        Return a string representation of the inverted index.

        Returns
        -------
        str
            A string representation of the inverted index.
        """
        return str(self.inverted_index)

    def print_line(self, word: str):
        """
        Print the line of the inverted index for a given word.

        Parameters
        ----------
        word: str
            The word to print the line for.
        """
        print(f"{word}: {self.inverted_index[word]}")

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
            print(f"Word {word} not in vocabulary")
            return None

        word_index = []
        parts = listdir(self.index_path)
        for i in range(len(parts)):
            partial_index = self.read_index(i)
            if word in partial_index:
                word_index.append(partial_index[word])

        sorted_word_index = dict(
            sorted(
                self.merge_inverted_index(word_index).items(),
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
            json.dump(inverted_index, f, indent=4)

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


if __name__ == "__main__":
    mem = [get_memory_usage()]
    args = parse_args("indexer")

    indexer = Indexer(args.Memory, args.Corpus, args.Index, 100, 46500)
    mem.append(get_memory_usage())

    tokenizer = indexer.tokenizer
    vocabulary = indexer.vocabulary

    indexer.parse_documents()
    mem.append(get_memory_usage())

    vocabulary.write_vocabulary()
    mem.append(get_memory_usage())

    for i in range(len(mem)):
        print(f"Memory Usage after step {i}: {mem[i]}")

    print(f"Vocabulary Size: {len(vocabulary)} unique tokens")
    print(f"Total Tokens: {vocabulary.total_count} tokens")
