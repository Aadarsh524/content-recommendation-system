# src/recommenders/item_similarity.py

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class ItemSimilarity:
    def __init__(self, path=None):
        self.sim_matrix = None
        self.items = None
        if path:
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.sim_matrix = data["sim_matrix"]
                self.items = data["items"]

    @staticmethod
    def build(train_df, user2idx, item2idx):
        # Create user-item sparse matrix
        rows = train_df['user_id'].map(user2idx)
        cols = train_df['item_id'].map(item2idx)
        data = np.ones(len(train_df))
        user_item = csr_matrix((data, (rows, cols)), shape=(len(user2idx), len(item2idx)))
        # Compute cosine similarity between items
        sim = cosine_similarity(user_item.T)  # items x items
        return {"sim_matrix": sim, "items": list(item2idx.keys())}

    def top_k(self, item_id, k=10):
        if self.sim_matrix is None or item_id not in self.items:
            return []
        idx = self.items.index(item_id)
        scores = self.sim_matrix[idx]
        top_idx = np.argsort(scores)[::-1][1:k+1]  # skip self
        return [self.items[i] for i in top_idx]
