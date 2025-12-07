import pickle
from scipy.sparse import csr_matrix

# Load your mappings and data
with open("models/als_maps.pkl", "rb") as f:
    u_to_idx, i_to_idx, users, items = pickle.load(f)

with open("models/train_valid_test.pkl", "rb") as f:
    train_df, valid_df, test_df = pickle.load(f)

# Build CSR: rows=users, cols=items
rows = train_df['user_id'].map(u_to_idx)
cols = train_df['item_id'].map(i_to_idx)
data = train_df['rating'].astype(float).values
user_item_csr = csr_matrix((data, (rows, cols)), shape=(len(u_to_idx), len(i_to_idx)))

# Overwrite previous possibly-wrong file!
with open("models/user_item_csr.pkl", "wb") as f:
    pickle.dump(user_item_csr, f)

print("user_item_csr.shape =", user_item_csr.shape)