import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, mlp_layers=[64,32], train_user_pos=None, all_items=None, device="cpu"):
        super().__init__()
        self.device = device
        self.n_users = n_users
        self.n_items = n_items
        self.all_items = all_items
        self.train_user_pos = train_user_pos  # dict: user_id -> set of seen items

        # GMF embeddings
        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)
        # MLP embeddings
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)
        # MLP layers
        mlp_input = emb_dim * 2
        layers = []
        for h in mlp_layers:
            layers.append(nn.Linear(mlp_input, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            mlp_input = h
        self.mlp = nn.Sequential(*layers)
        # Final layer
        final_size = emb_dim + (mlp_layers[-1] if len(mlp_layers) > 0 else emb_dim)
        self.output = nn.Linear(final_size, 1)
        self.sig = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gmf_user.weight, std=0.01)
        nn.init.normal_(self.gmf_item.weight, std=0.01)
        nn.init.normal_(self.mlp_user.weight, std=0.01)
        nn.init.normal_(self.mlp_item.weight, std=0.01)

    def forward(self, u, i):
        u = u.squeeze(1); i = i.squeeze(1)
        g_u = self.gmf_user(u); g_i = self.gmf_item(i)
        gmf = g_u * g_i
        m_u = self.mlp_user(u); m_i = self.mlp_item(i)
        mlp = torch.cat([m_u, m_i], dim=1)
        mlp_out = self.mlp(mlp)
        x = torch.cat([gmf, mlp_out], dim=1)
        out = self.sig(self.output(x)).unsqueeze(1)
        return out

    def score_all(self, user_id, user2idx, idx2item, topk=10, filter_seen=True):
        """Return top-k recommended item IDs for a user"""
        self.eval()
        if user_id not in user2idx:
            return []

        u_idx = user2idx[user_id]
        user_tensor = torch.LongTensor([[u_idx]]).to(self.device)
        batch_size = 2048
        scores = []
        item_indices = list(range(self.n_items))
        with torch.no_grad():
            for i in range(0, self.n_items, batch_size):
                items_batch = item_indices[i:i+batch_size]
                it_tensor = torch.LongTensor([[j] for j in items_batch]).to(self.device)
                u_batch = user_tensor.repeat(len(it_tensor),1)
                out = self.forward(u_batch, it_tensor)
                scores.extend(out.squeeze(1).cpu().numpy().tolist())

        scored = list(zip(self.all_items, scores))
        if filter_seen and self.train_user_pos is not None:
            seen = self.train_user_pos.get(user_id, set())
            scored = [(it,sc) for it,sc in scored if it not in seen]

        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)[:topk]
        return [it for it,_ in scored_sorted]
