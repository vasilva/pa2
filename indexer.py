from parser import Parser
from tokenizer import Tokenizer
from util import get_memory_usage, parse_args


class Indexer:
    """
    Class to handle indexing of a directory of files.
    """

    def __init__(self, memory: int, corpus: str, index: str, max_docs: int = 10):
        self.memory = memory
        self.corpus = corpus
        self.index = index
        self.inverted_index = dict()
        self.parser = Parser(corpus, max_docs)
        self.tokenizer = Tokenizer()

    def parse_files(self):
        """
        Parse the corpus file and create an inverted index.

        TODO: Make it multithreaded.
        """
        self.parser.parse_corpus()
        for doc_id, doc in self.parser.documents.items():
            self.add_to_inverted_index(doc_id, doc)

    def add_to_inverted_index(self, doc_id: int, doc):
        """
        Add the document to the inverted index.
        """
        # Tokenize the title, text, and keywords
        document = doc["title"] + " " + doc["text"] + " " + " ".join(doc["keywords"])
        tokens = self.tokenizer.tokenize_text(document)
        # Add the tokens to the vocabulary
        self.tokenizer.add_to_vocabulary(tokens)

        # Add the tokens to the inverted index
        for token in tokens:
            if token not in self.inverted_index:
                self.inverted_index[token] = {doc_id: 1}
            else:
                self.inverted_index[token][doc_id] = (
                    self.inverted_index[token].get(doc_id, 0) + 1
                )

    def __str__(self):
        """
        Return a string representation of the inverted index.
        """
        return str(self.inverted_index)


if __name__ == "__main__":
    mem = [get_memory_usage()]
    args = parse_args()
    indexer = Indexer(args.Memory, args.Corpus, args.Index, 1000)
    max_memory = args.Memory * 1024 * 1024
    mem.append(get_memory_usage())
    indexer.parse_files()
    mem.append(get_memory_usage())
    for i, m in enumerate(mem):
        print(f"Memory Usage after step {i}: {m}")

    print(indexer.inverted_index)
    print(f"Memory Usage: {get_memory_usage()/ (1024 ** 2):.2f} MB")
    print(f"Vocabulary Size: {len(indexer.tokenizer)} unique tokens")
    print(f"Total Tokens: {indexer.tokenizer.total_count} tokens")
