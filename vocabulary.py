from json import dump, load


class Vocabulary:
    """
    Class to handle vocabulary operations.
    """

    def __init__(self, file_path: str = "data/vocabulary.json"):
        """
        Initialize the Vocabulary with an empty dictionary.

        Parameters
        ----------
        file_path: str
            The path to the vocabulary file.
        """
        self.vocabulary = dict()
        self.total_count = 0
        self.file_path = file_path

    def __len__(self):
        return len(self.vocabulary)

    def __bool__(self):
        return len(self.vocabulary) > 0

    def __str__(self):
        return str(self.vocabulary)

    def __contains__(self, token: str):
        return token in self.vocabulary

    def __getitem__(self, word):
        return self.vocabulary[word] if word in self.vocabulary else 0

    def add(self, tokens: list[str]):
        """
        Add tokens to the vocabulary.

        Parameters
        ----------
        tokens: list[str]
            The list of tokens to be added to the vocabulary.
        """
        for token in tokens:
            # Add new token or increment its count
            if token not in self.vocabulary:
                self.vocabulary[token] = 1

            else:
                self.vocabulary[token] += 1

            self.total_count += 1

    def read_vocabulary(self):
        """
        Read the vocabulary from the json file in the specified path.

        Returns
        -------
        bool:
            True if the vocabulary was read successfully, False otherwise.
        """
        with open(self.file_path, "r") as f:
            self.vocabulary = load(f)
            self.total_count = sum(self.vocabulary.values())

        return self.vocabulary is not None

    def write_vocabulary(self):
        """
        Write the vocabulary to the json file in the specified path.
        """
        with open(self.file_path, "w") as f:
            dump(dict(sorted(self.vocabulary.items())), f, indent=2)

    def merge(self, other: "Vocabulary"):
        """
        Merge another vocabulary into this one.

        Parameters
        ----------
        other: Vocabulary
            The other vocabulary to be merged.
        """
        for token, count in other.vocabulary.items():
            if token not in self.vocabulary:
                self.vocabulary[token] = count
            else:
                self.vocabulary[token] += count

        self.total_count += other.total_count
