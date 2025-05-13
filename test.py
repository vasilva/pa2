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

import json
from re import sub

stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
with open("corpus.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i == 92:
            text = json.loads(line)
            print(text, end="\n\n")

            tokens = word_tokenize(text["text"])
            print(tokens, end="\n\n")
            stemmed_tokens = []
            for token in tokens:
                stemmed_word = token.replace("/", " ")
                stemmed_word = stemmed_word.replace("\\", " ")
                stemmed_word = stemmed_word.replace("-", " ")
                stemmed_word = stemmed_word.replace("–", " ")
                if "." in stemmed_word and len(stemmed_word) > 1:
                    stemmed_word = stemmed_word.replace(".", " ")
                if "," in stemmed_word and len(stemmed_word) > 1:
                    stemmed_word = stemmed_word.replace(",", " ")
                if ":" in stemmed_word and len(stemmed_word) > 1:
                    stemmed_word = stemmed_word.replace(":", " ")
                if ";" in stemmed_word and len(stemmed_word) > 1:
                    stemmed_word = stemmed_word.replace(";", " ")
                
                stemmed_tokens.extend(stemmed_word.split())

            print(stemmed_tokens, end="\n\n")
            stemmed_tokens = [
                stemmer.stem(token)
                for token in stemmed_tokens
                if token not in stop_words
            ]
            print(stemmed_tokens)
            break
