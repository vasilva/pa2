# Download NLTK resources if not already downloaded
from nltk import download

download("punkt", quiet=True)
download("punkt_tab", quiet=True)
download("stopwords", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from re import sub


class Tokenizer:
    """
    Class to handle tokenization of text data.
    """

    def __init__(self, language: str = "english"):
        """
        Initialize the Tokenizer with NLTK resources.
        """
        self.stop_words = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language)

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

        TODO: Call it in the indexer.py file, multithreaded
        """
        words = word_tokenize(text)
        words_filtered = []
        for word in words:
            stemmed_words = []
            if word.lower() not in self.stop_words:
                stemmed_word = self.stemmer.stem(word)
                stemmed_word = sub("([\\'\".?,-/])", r" \1", stemmed_word)
                stemmed_word = stemmed_word.replace("/", " ")
                stemmed_word = stemmed_word.replace("\\", " ")
                stemmed_word = stemmed_word.replace("-", " ")
                stemmed_word = stemmed_word.replace("–", " ")
                stemmed_words.extend(stemmed_word.split())

            words_filtered.extend(stemmed_words)

        return words_filtered
