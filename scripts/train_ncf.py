import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pickle
from pathlib import Path

from src.models.ncf import NCF
from src.models.bpr_dataset import BPRDataset
from src.data.load_data import load_train_val_test
from src.evaluation.evaluate import evaluate_model
from src.models.baseline import train_popularity_model
from src.config import TRAINED_MODEL


import torch
import torch.nn.functional as F

def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))


def build_user_items(df):
    return df.groupby("user_id")["item_id"].apply(set).to_dict()


def encode_ids(df):
    user2idx = {u: i for i, u in enumerate(df["user_id"].unique())}
    item2idx = {i: j for j, i in enumerate(df["item_id"].unique())}

    df = df.copy()
    df["user_id"] = df["user_id"].map(user2idx)
    df["item_id"] = df["item_id"].map(item2idx)
    return df, user2idx, item2idx


def train_bpr_ncf(
    epochs=50,
    embedding_dim=64,
    hidden_layers=[128, 64],
    lr=1e-3,
    batch_size=1024,
    device="cpu",
):
    train, _, test = load_train_val_test()

    train, user2idx, item2idx = encode_ids(train)
    test = test.copy()
    test["user_id"] = test["user_id"].map(user2idx)
    test["item_id"] = test["item_id"].map(item2idx)
    test = test.dropna().astype(int)

    train_user_items = build_user_items(train)
    test_user_items = build_user_items(test)

    n_users = len(user2idx)
    n_items = len(item2idx)

    dataset = BPRDataset(train_user_items, n_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NCF(
        n_users,
        n_items,
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for u, i_pos, i_neg in loader:
            u = u.to(device)
            i_pos = i_pos.to(device)
            i_neg = i_neg.to(device)

            optimizer.zero_grad()

            pos_scores = model(u, i_pos)
            neg_scores = model(u, i_neg)

            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, BPR Loss: {total_loss/len(loader):.4f}")

    # Save artifacts
    Path(TRAINED_MODEL).mkdir(parents=True, exist_ok=True)
    with open(f"{TRAINED_MODEL}/ncf_bpr.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "user2idx": user2idx,
                "item2idx": item2idx,
                "n_items": n_items,
                "train_user_items": train_user_items,
            },
            f,
        )

    popular_items = train_popularity_model(train)

    def recommend_fn(user_id, seen_items, k=10):
        if user_id not in train_user_items:
            return popular_items[:k]

        model.eval()
        candidates = [
            i for i in range(n_items) if i not in seen_items
        ]
        user_tensor = torch.tensor([user_id] * len(candidates)).to(device)
        item_tensor = torch.tensor(candidates).to(device)

        with torch.no_grad():
            scores = model(user_tensor, item_tensor)

        topk_idx = torch.topk(scores, k=min(k, len(scores))).indices
        return [candidates[i] for i in topk_idx.cpu().numpy()]

    results = evaluate_model(
        recommend_fn,
        test_user_items,
        train_user_items,
        k=10,
    )

    print("🔥 BPR-NCF Results:", results)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_bpr_ncf(
        epochs=50,
        embedding_dim=64,
        hidden_layers=[128, 64],
        device=device,
    )
