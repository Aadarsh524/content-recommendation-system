import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from src.models.ncf import NCF
from src.evaluation.evaluate import evaluate_model
from src.models.baseline import train_popularity_model
from src.data.load_data import load_train_val_test

# ============================
# Dataset with negative sampling
# ============================
class NCFDataset(Dataset):
    def __init__(self, user_item_dict, n_items, neg_samples=32):
        self.user_item_dict = user_item_dict
        self.users = list(user_item_dict.keys())
        self.n_items = n_items
        self.neg_samples = neg_samples
        self.data = []
        self.prepare_data()

    def prepare_data(self):
        for u, pos_items in self.user_item_dict.items():
            for i in pos_items:
                self.data.append((u, i, 1))
                # negative sampling
                neg_candidates = [j for j in range(self.n_items) if j not in pos_items]
                if len(neg_candidates) == 0:
                    continue
                neg_items = np.random.choice(
                    neg_candidates,
                    size=min(self.neg_samples, len(neg_candidates)),
                    replace=len(neg_candidates) < self.neg_samples
                )
                for j in neg_items:
                    self.data.append((u, j, 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, r = self.data[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long), torch.tensor(r, dtype=torch.float)

# ============================
# Encode IDs to 0-index
# ============================
def encode_ids(df):
    user2idx = {u: i for i, u in enumerate(df['user_id'].unique())}
    item2idx = {i: j for j, i in enumerate(df['item_id'].unique())}
    df = df.copy()
    df['user_id'] = df['user_id'].map(user2idx)
    df['item_id'] = df['item_id'].map(item2idx)
    return df, user2idx, item2idx

def build_user_items(df):
    return df.groupby("user_id")["item_id"].apply(set).to_dict()

# ============================
# Train NCF
# ============================
def train_ncf_model(train_user_items, n_users, n_items, embedding_dim=32, hidden_layers=[64,32,16], epochs=5, lr=0.001, device='cpu'):
    dataset = NCFDataset(train_user_items, n_items)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = NCF(n_users, n_items, embedding_dim=embedding_dim, hidden_layers=hidden_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, r in loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            optimizer.zero_grad()
            pred = model(u, i)
            loss = criterion(pred, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    return model

# ============================
# Recommendation function
# ============================
def recommend_ncf(model, user_id, seen_items, n_items, topk=10, device='cpu'):
    model.eval()
    all_items = torch.tensor([i for i in range(n_items) if i not in seen_items], dtype=torch.long).to(device)
    if len(all_items) == 0:
        return []
    user = torch.tensor([user_id]*len(all_items), dtype=torch.long).to(device)
    with torch.no_grad():
        scores = model(user, all_items)
    topk_idx = torch.topk(scores, k=min(topk, len(scores)))[1]
    recommended = all_items[topk_idx].cpu().numpy()
    return recommended.tolist()

# ============================
# Full training pipeline
# ============================
def train_ncf(epochs=5, embedding_dim=64, hidden_layers=[64,32,16], device='cpu'):
    train, val, test = load_train_val_test()

    # Encode train
    train, user2idx, item2idx = encode_ids(train)

    # Map test safely, drop unknown users/items
    test = test.copy()
    test['user_id'] = test['user_id'].map(user2idx)
    test['item_id'] = test['item_id'].map(item2idx)
    test = test.dropna(subset=['user_id','item_id'])
    test['user_id'] = test['user_id'].astype(int)
    test['item_id'] = test['item_id'].astype(int)

    train_user_items = build_user_items(train)
    test_user_items = build_user_items(test)

    n_users = len(user2idx)
    n_items = len(item2idx)

    model = train_ncf_model(train_user_items, n_users, n_items,
                            embedding_dim=embedding_dim,
                            hidden_layers=hidden_layers,
                            epochs=epochs,
                            device=device)

    # Popular items for cold-start
    popular_items = train_popularity_model(train)  # expects DataFrame

    # Evaluation
    def recommend_fn(user_id, seen_items, k=10):
        if user_id not in train_user_items:
            return popular_items[:k]
        return recommend_ncf(model, user_id, seen_items, n_items, topk=k, device=device)

    results = evaluate_model(recommend_fn, test_user_items, train_user_items, k=10)
    print("NCF Results:", results)

# ============================
# Run
# ============================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Final training: 50 epochs, larger embeddings
    train_ncf(epochs=100, embedding_dim=256, hidden_layers=[256,128,64], device=device)
