import json


class Parser:
    """
    Class to handle parsing of a corpus file.
    """

    def __init__(self, corpus: str, max_lines: int = 10):
        """
        Initialize the Parser with the corpus file and maximum lines to parse.

        Parameters
        ----------
        corpus : str
            Path to the corpus file.
        max_lines : int
            Maximum number of lines to parse from the corpus file.
        """
        self.corpus = corpus
        self.max_lines = max_lines
        self.documents = dict()

    def parse_corpus(self):
        """
        Parse the jsonl corpus file and return a list of documents.
        """
        with open(self.corpus, "r") as file:
            for i, line in enumerate(file):
                if i >= self.max_lines:
                    break
                doc = json.loads(line)
                doc = {
                    "id": int(doc["id"]),
                    "title": doc["title"],
                    "text": doc["text"],
                    "keywords": doc["keywords"],
                }
                self.documents[doc["id"]] = doc

    def print_documents(self, num_docs: int = 10):
        """
        Print the parsed documents.
        """
        if not self.documents:
            print("No documents parsed yet")

        for i, (doc_id, doc) in enumerate(self.documents.items()):
            if i >= num_docs:
                break

            print(f"Document ID: {doc_id}")
            print(f"Title: {doc['title']}")
            print(f"Text: {doc['text']}")
            print(f"Keywords: {doc['keywords']}")

        print(f"Total Documents: {len(self.documents)}")
