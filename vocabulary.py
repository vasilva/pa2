from json import dump, load


class Vocabulary:
    """
    Class to handle vocabulary operations.
    """

    def __init__(self, file_path: str = "vocabulary.json"):
        """
        Initialize the Vocabulary with an empty dictionary.
        """
        self.vocabulary = dict()
        self.total_count = 0
        self.file_path = file_path

    def __len__(self):
        """
        Return the size of the vocabulary.

        Returns
        -------
        int
            The size of the vocabulary.
        """
        return len(self.vocabulary)

    def __str__(self):
        """
        Return a string representation of the vocabulary.

        Returns
        -------
        str
            A string representation of the vocabulary.
        """
        return str(self.vocabulary)

    def add(self, tokens: list[str]):
        """
        Add tokens to the vocabulary.

        Parameters
        ----------
        tokens: list[str]
            The list of tokens to be added to the vocabulary.
        """
        for token in tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = 1

            else:
                self.vocabulary[token] += 1

            self.total_count += 1

    def read_vocabulary(self):
        """
        Read the vocabulary from a file.
        """
        with open(self.file_path, "r") as f:
            self.vocabulary = load(f)
            self.total_count = sum(self.vocabulary.values())
            return True

        return False

    def write_vocabulary(self):
        """
        Write the vocabulary to a file.
        """
        with open(self.file_path, "w") as f:
            dump(self.vocabulary, f, indent=4)