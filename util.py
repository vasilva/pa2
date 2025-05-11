import psutil
from argparse import ArgumentParser


def get_memory_usage():
    """
    Get the current memory usage of the process.

    Returns
    -------
        int: The current memory usage in bytes.
    """
    return psutil.Process().memory_info().rss


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = ArgumentParser(
        description="Index a directory of files.",
        usage="python3 indexer.py -m <int> -c <str> -i <str>",
    )
    parser.add_argument(
        "-m",
        "--Memory",
        type=int,
        required=False,
        help="The memory available to the indexer in megabytes.",
        default=1024,
    )
    parser.add_argument(
        "-c",
        "--Corpus",
        type=str,
        required=False,
        help="The path to the corpus file to be indexed.",
        default="corpus.jsonl",
    )
    parser.add_argument(
        "-i",
        "--Index",
        type=str,
        required=False,
        help="The path to the directory where indexes should be written.",
        default="index.json",
    )
    return parser.parse_args()


def term_frequency(vocabulary: dict, token: str, total_count: int) -> float:
    """
    Get the term frequency of a token.

    Parameters
    ----------
    vocabulary : dict
        The vocabulary dictionary containing token counts.
    token : str
        The token for which to calculate the term frequency.
    total_count : int
        The total count of tokens in the vocabulary.

    Returns
    -------
    float
        The term frequency of the token.

    TODO: Call it in query.py file, after indexing
    """
    return (vocabulary[token] / total_count) if token in vocabulary else 0.0
