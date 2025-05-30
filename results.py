from util import *


def get_ranks(ranker_file):
    ranks = {}
    with open(ranker_file, "r") as f:
        for line in f:
            query = line.split("\n")[0]
            ranks[query] = []
            for _ in range(10):
                _, score = f.readline().split()
                ranks[query].append(int(score))
    return ranks


ranks_bm25 = get_ranks("bm25.txt")
ranks_tfidf = get_ranks("tfidf.txt")
ranks_plnvsm = get_ranks("plnvsm.txt")

for query, rank in ranks_bm25.items():
    print(query)
    ndcg = nDCG(rank)
    print(ndcg)
