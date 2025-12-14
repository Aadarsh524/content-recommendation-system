import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


class ALSRecommender:
    def __init__(self, factors=128, regularization=0.1, iterations=20):
        # ✅ MODEL MUST BE INITIALISED HERE
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations
        )

        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.user_item_matrix = None

    def fit(self, ratings_df):
        # Create ID mappings
        self.user_map = {u: i for i, u in enumerate(ratings_df.user_id.unique())}
        self.item_map = {i: j for j, i in enumerate(ratings_df.item_id.unique())}

        self.reverse_user_map = {i: u for u, i in self.user_map.items()}
        self.reverse_item_map = {j: i for i, j in self.item_map.items()}

        ratings_df = ratings_df.copy()
        ratings_df["user_idx"] = ratings_df.user_id.map(self.user_map)
        ratings_df["item_idx"] = ratings_df.item_id.map(self.item_map)

        self.user_item_matrix = csr_matrix(
            (
                ratings_df.rating,
                (ratings_df.user_idx, ratings_df.item_idx)
            ),
            shape=(len(self.user_map), len(self.item_map))
        )

        # ✅ NOW THIS WILL WORK
        self.model.fit(self.user_item_matrix)

        print("ALS matrix shape:", self.user_item_matrix.shape)

    def recommend(self, user_id, seen_items=None, N=10, popular_items=None):
        if user_id not in self.user_map:
            return popular_items[:N] if popular_items is not None else []

        user_idx = self.user_map[user_id]

        item_ids, scores = self.model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=N
        )

        return [self.reverse_item_map[i] for i in item_ids]
