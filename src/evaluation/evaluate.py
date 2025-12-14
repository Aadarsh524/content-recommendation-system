import numpy as np
from src.metrics.ranking import precision_at_k, recall_at_k, ndcg_at_k

def evaluate_model(
    recommend_fn,
    users_relevant_items,
    users_seen_items,
    k=10
):
    precisions, recalls, ndcgs = [], [], []

    for user_id, relevant_items in users_relevant_items.items():
        seen_items = users_seen_items.get(user_id, set())

        recommendations = recommend_fn(user_id, seen_items)

        if not recommendations:
            continue

        precisions.append(
            precision_at_k(recommendations, relevant_items, k)
        )
        recalls.append(
            recall_at_k(recommendations, relevant_items, k)
        )
        ndcgs.append(
            ndcg_at_k(recommendations, relevant_items, k)
        )

    return {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "ndcg": np.mean(ndcgs),
        "users": len(precisions)
    }
