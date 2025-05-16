# Download NLTK resources if not already downloaded
from nltk import download
from nltk.data import find

try:
    find("tokenizers/punkt")
except LookupError:
    download("punkt")

try:
    find("tokenizers/punkt_tab")
except LookupError:
    download("punkt_tab")

try:
    find("corpora/stopwords")
except LookupError:
    download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class Tokenizer:
    """
    Class to handle tokenization of text data.
    """

    def __init__(self, language: str = "english"):
        """
        Initialize the Tokenizer with NLTK resources.

        Parameters
        ----------
        language : str
            The language to be used for tokenization. Default is "english".
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language)

    def __len__(self):
        return len(self.vocabulary)

    def __str__(self):
        return str(self.vocabulary)

    def tokenize_text(self, text: str):
        """
        Tokenize the input text and return a list of tokens.


        Parameters
        ----------
        text : str
            The input text to be tokenized.

        Returns
        -------
        list[str]
            A list of tokens extracted from the input text.
        """
        words = word_tokenize(text, self.language)
        filtered_words, stemmed_words = [], []
        for word in words:
            # Remove and separate words with '/\-_' characteres
            word = word.replace("/", " ")
            word = word.replace("\\", " ")
            word = word.replace("-", " ")
            word = word.replace("–", " ")

            # Remove if it's between two words
            if "." in word and len(word) > 1:
                word = word.replace(".", " ")
            if "," in word and len(word) > 1:
                word = word.replace(",", " ")
            if ":" in word and len(word) > 1:
                word = word.replace(":", " ")
            if ";" in word and len(word) > 1:
                word = word.replace(";", " ")

            filtered_words.extend(word.split())

        # Stem and stopwords
        stemmed_words = [
            self.stemmer.stem(word)
            for word in filtered_words
            if word not in self.stop_words
        ]

        return stemmed_words
