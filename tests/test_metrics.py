from src.metrics.ranking import precision_at_k, recall_at_k, ndcg_at_k

recommended = [10, 20, 30, 40, 50]
relevant = {20, 50, 70}

print("Precision:", precision_at_k(recommended, relevant, 5))
print("Recall:", recall_at_k(recommended, relevant, 5))
print("NDCG:", ndcg_at_k(recommended, relevant, 5))
