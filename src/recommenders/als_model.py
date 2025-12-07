import pickle

class ALSWrapper:
    def __init__(self, model_path="models/als_model.pkl", user_item_csr=None, u_to_idx=None, i_to_idx=None):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.user_item_csr = user_item_csr  # sparse CSR matrix of user-item
        self.u_to_idx = u_to_idx
        self.i_to_idx = i_to_idx
        # reverse mapping
        self.idx_to_item = {v:k for k,v in i_to_idx.items()} if i_to_idx is not None else None

    def recommend(self, user_id, k=10):
        if user_id not in self.u_to_idx:
            return []
        uid = self.u_to_idx[user_id]
        recs = self.model.recommend(uid, self.user_item_csr, N=k, filter_already_liked_items=True)
        return [self.idx_to_item[iid] for iid,_ in recs]
