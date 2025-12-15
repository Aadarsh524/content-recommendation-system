import torch
from torch.utils.data import Dataset
import random

class BPRDataset(Dataset):
    def __init__(self, user_item_dict, n_items):
        self.user_item_dict = user_item_dict
        self.users = list(user_item_dict.keys())
        self.n_items = n_items
        self.samples = []
        self._prepare()

    def _prepare(self):
        for u, pos_items in self.user_item_dict.items():
            for i in pos_items:
                self.samples.append((u, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, i_pos = self.samples[idx]

        # sample negative
        while True:
            i_neg = random.randint(0, self.n_items - 1)
            if i_neg not in self.user_item_dict[u]:
                break

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i_pos, dtype=torch.long),
            torch.tensor(i_neg, dtype=torch.long),
        )
