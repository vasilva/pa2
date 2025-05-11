from tokenizer import Tokenizer
from vocabulary import Vocabulary

import json
from util import get_memory_usage, parse_args, term_frequency as tf

import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count, makedirs, path

n_threads = cpu_count() - 6

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
        memory: int,
        corpus: str,
        index: str,
        batch_size: int = 10,
        max_doc: int = 987,
    ):
        self.memory = memory
        self.corpus = corpus
        self.index = index
        self.inverted_index = dict()
        self.tokenizer = Tokenizer()
        self.vocabulary = Vocabulary()
        self.batch_size = batch_size
        self.max_doc = max_doc

    def parse_documents(self):
        """
        Parse the corpus file and create an inverted index.

        TODO: Make it multithreaded.
        """
        with open(self.corpus, "r", encoding="utf8") as f:
            data = []
            # Read the file line by line
            for i, line in enumerate(f):
                if self.max_doc < 0:
                    # If max_doc is negative, process all documents
                    pass
                elif i >= self.max_doc:
                    # If max_doc is reached, stop processing
                    break

                # Read the file in batches
                doc = json.loads(line)
                data.append(doc)

                if len(data) >= self.batch_size or i == self.max_doc - 1:
                    # If the batch is full, process it
                    with ThreadPoolExecutor(max_workers=len(data)) as executor:
                        futures = [
                            executor.submit(self.make_partial_inverted_index, doc)
                            for doc in data
                        ]

                    # Wait for all futures to complete
                    for future in as_completed(futures):
                        try:
                            partial_inverted_index = future.result()
                            self.merge_inverted_index(partial_inverted_index)

                        except Exception as e:
                            print(f"Error processing document: {e}")
                    # Clear the batch
                    data = []
                    print(f"Processed {i + 1} documents")

    def make_partial_inverted_index(self, doc):
        """
        Add the document to the inverted index.
        """
        # Tokenize the title, text, and keywords
        doc_id = int(doc["id"])
        document = doc["title"] + " " + doc["text"] + " " + " ".join(doc["keywords"])
        tokens = self.tokenizer.tokenize_text(document)
        inverted_index = dict()
        # Add the tokens to the vocabulary
        self.vocabulary.add(tokens)

        # Add the tokens to the inverted index
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = {doc_id: 1}
            else:
                inverted_index[token][doc_id] = inverted_index[token].get(doc_id, 0) + 1

        return inverted_index

    def merge_inverted_index(self, inverted_index):
        """
        Merge the inverted index with the existing inverted index.
        """
        for token, doc_ids in inverted_index.items():
            if token not in self.inverted_index:
                self.inverted_index[token] = doc_ids
            else:
                for doc_id, count in doc_ids.items():
                    if doc_id not in self.inverted_index[token]:
                        self.inverted_index[token][doc_id] = count
                    else:
                        self.inverted_index[token][doc_id] += count

    def __str__(self):
        """
        Return a string representation of the inverted index.
        """
        return str(self.inverted_index)

    def print_line(self, word: str):
        """
        Print the line of the inverted index for a given word.

        Parameters
        ----------
        word : str
            The word to print the line for.
        """
        print(f"{word}: {self.inverted_index[word]}")

    def write_index(self):
        """
        Write the inverted index to a file.
        """
        with open(self.index, "w") as f:
            json.dump(self.inverted_index, f, indent=4)

    def read_index(self):
        """
        Read the inverted index from a file.
        """
        with open(self.index, "r") as f:
            self.inverted_index = json.load(f)


if __name__ == "__main__":
    mem = [get_memory_usage()]
    args = parse_args()

    indexer = Indexer(args.Memory, args.Corpus, args.Index, 100)
    mem.append(get_memory_usage())

    tokenizer = indexer.tokenizer
    vocabulary = indexer.vocabulary

    indexer.parse_documents()
    mem.append(get_memory_usage())

    indexer.write_index()
    mem.append(get_memory_usage())

    vocabulary.write_vocabulary()
    mem.append(get_memory_usage())

    for i in range(len(mem)):
        print(f"Memory Usage after step {i}: {mem[i]}")

    print(f"Vocabulary Size: {len(vocabulary)} unique tokens")
    print(f"Total Tokens: {vocabulary.total_count} tokens")
