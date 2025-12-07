import pickle
import torch
import numpy as np
from scipy.sparse import csr_matrix
from src.recommenders.nfc_model import NCF


class RecommenderService:
    def __init__(self, device="cpu"):
        self.device = device
        
        # 1. Load Mappings
        print("Loading NCF metadata...")
        with open("models/ncf_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.user2idx = meta["user2idx"]
        self.item2idx = meta["item2idx"]
        self.all_users = meta["all_users"]
        self.all_items = meta["all_items"]

        # 2. Load NCF Model
        print("Loading NCF model...")
        n_users = len(self.all_users)
        n_items = len(self.all_items)
        # Make sure NCF class matches the one used during training
        self.ncf = NCF(n_users, n_items) 
        self.ncf.load_state_dict(torch.load("models/ncf_state.pt", map_location=device))
        self.ncf.to(device)
        self.ncf.eval()

        # 3. Load ALS Model & Matrix
        # print("Loading ALS model...")
        # with open("models/als_model.pkl", "rb") as f:
        #     self.als = pickle.load(f)
            
        # with open("models/user_item_csr.pkl", "rb") as f:
        #     self.user_item_csr = pickle.load(f)
        
        # # Ensure matrix is CSR for fast slicing
        # if not isinstance(self.user_item_csr, csr_matrix):
        #     self.user_item_csr = csr_matrix(self.user_item_csr)

        # 4. Load Similarity Model (Optional fallback)
        print("Loading Similarity model...")
        try:
            with open("models/item_similarity.pkl", "rb") as f:
                self.sim = pickle.load(f)
        except FileNotFoundError:
            print("Warning: item_similarity.pkl not found. Fallbacks may fail.")
            self.sim = None

    def recommend(self, user_id, k=10):
        """ ALS-based Recommendation """
        # Handle New User / Cold Start
        if user_id not in self.user2idx:
            if self.sim and hasattr(self.sim, "top_k_popular"):
                return self.sim.top_k_popular(k)
            return []

        uid = self.user2idx[user_id]

        # FIX: Pass scalar 'uid' and the full 'user_items' matrix
        # filter_already_liked_items=True automatically checks the matrix
        ids, scores = self.als.recommend(
            userid=uid,
            user_items=self.user_item_csr,
            N=k,
            filter_already_liked_items=True
        )

        # Map internal indices back to Raw IDs
        return [self.all_items[i] for i in ids]

    def recommend_neural(self, user_id, k=10):
        """ Neural Collaborative Filtering Recommendation """
        if user_id not in self.user2idx:
            if self.sim and hasattr(self.sim, "top_k_popular"):
                return self.sim.top_k_popular(k)
            return []
            
        u_idx = self.user2idx[user_id]
        user_tensor = torch.LongTensor([[u_idx]]).to(self.device)

        # --- 1. Identify "Seen" items (History) ---
        # Get the row from the sparse matrix corresponding to this user
        # .indices returns the INTERNAL ITEM INDICES the user interacted with
        user_row = self.user_item_csr.getrow(u_idx)
        seen_indices = set(user_row.indices)

        # --- 2. Score all items in batches ---
        batch_size = 2048
        scores = []
        item_indices = list(range(len(self.all_items)))
        
        with torch.no_grad():
            for i in range(0, len(item_indices), batch_size):
                batch_items = item_indices[i:i+batch_size]
                item_tensor = torch.LongTensor([[j] for j in batch_items]).to(self.device)
                
                # Repeat user tensor to match item batch size
                u_batch = user_tensor.repeat(len(item_tensor), 1)
                
                out = self.ncf(u_batch, item_tensor)
                scores.extend(out.squeeze(1).cpu().numpy().tolist())

        # --- 3. Filter and Sort ---
        # We zip indices with scores, filter out seen indices, then map to Raw IDs
        # (Faster than zipping Raw IDs first)
        
        candidates = []
        for i, score in enumerate(scores):
            if i not in seen_indices:
                candidates.append((i, score))
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take top K and convert to Raw ID
        top_k = candidates[:k]
        return [self.all_items[idx] for idx, score in top_k]

    def similar(self, item_id, k=10):
        if self.sim is None:
            return []
        # Ensure your ItemSimilarity class has 'top_k' method
        return self.sim.top_k(item_id, k)