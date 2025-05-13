import psutil
from argparse import ArgumentParser

import numpy as np


def get_memory_usage():
    """
    Get the current memory usage of the process.

    Returns
    -------
        int: The current memory usage in bytes.
    """
    return psutil.Process().memory_info().rss


def parse_args(program: str, required=False):
    """
    Parse command-line arguments.

    Parameters
    ----------
        program: str
            The program to get the arguments. `'indexer'` or `'processor'`.
        required: bool
            Whether the arguments are required or not.

    Returns
    -------
        argparse.Namespace: Parsed command-line arguments.
    """
    if program == "indexer":
        parser = ArgumentParser(
            description="Index a directory of files.",
            usage="python3 indexer.py -m <MEMORY> -c <CORPUS> -i <INDEX>",
        )
        parser.add_argument(
            "-m",
            "--Memory",
            type=int,
            required=required,
            help="The memory available to the indexer in megabytes.",
            default=1024,
        )
        parser.add_argument(
            "-c",
            "--Corpus",
            type=str,
            required=required,
            help="The path to the corpus file to be indexed.",
            default="corpus.jsonl",
        )

    elif program == "processor":
        parser = ArgumentParser(
            description="Process the queries.",
            usage="python3 processor.py -i <INDEX> -q <QUERIES> -r <RANKER>",
        )
        parser.add_argument(
            "-q",
            "--Query",
            type=str,
            required=required,
            help="The path to a file with the list of queries to process.",
            default="queries.txt",
        )
        parser.add_argument(
            "-r",
            "--Ranker",
            type=str,
            required=required,
            help="A string informing the ranking function (either 'TFIDF' or 'BM25') to be used to score documents for each query.",
            default="BM25",
        )

    else:
        raise ValueError("Program must be 'indexer' or 'processor'")

    parser.add_argument(
        "-i",
        "--Index",
        type=str,
        required=required,
        help="The path to the directory of the index files.",
        default="index",
    )
    return parser.parse_args()


def TF(word_count: int) -> float:
    """
    Get the term frequency of a token.

    Parameters
    ----------
    word_count: int
        The number of times the token appears in the document.

    Returns
    -------
    float
        The term frequency of the token.
    """
    return np.log(1 + word_count)


def IDF(n_total: int, n_word: int) -> float:
    """
    Get the inverse document frequency of a token.

    Parameters
    ----------
    n_total: int
        The total number of documents in the corpus.
    n_word: int
        The number of documents containing the token.

    Returns
    -------
    float
        The inverse document frequency of the token.
    """
    return np.log((n_total + 1) / n_word) if n_word > 0 else 0.0


def PLN(b: float, doc_len: int, avdl: float) -> float:
    """
    Get the pivoted length normalization of the document.

    Parameters
    ----------
    b: float
        Hiper-parameter [0,1]
    doc_len: int
        Document length in tokens.
    avdl: float
        Average document length in the corpus.

    Returns
    -------
    float
        The pivoted length normalization of the document.
    """
    return (1 - b) + b * doc_len / avdl
