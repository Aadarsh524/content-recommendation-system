import numpy as np

def precision_at_k(recommended, relevant, k):
    if k == 0:
        return 0.0
    recommended = recommended[:k]
    return len(set(recommended) & set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    recommended = recommended[:k]
    return len(set(recommended) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)

    idcg = sum(
        1 / np.log2(i + 2)
        for i in range(min(k, len(relevant)))
    )

    return dcg / idcg if idcg > 0 else 0.0
