import pickle
from src.recommenders.item_similarity import ItemSimilarity

# Load train_df
with open("models/train_valid_test.pkl","rb") as f:
    train_df, valid_df, test_df = pickle.load(f)

# Build user/item mapping
all_users = sorted(train_df['user_id'].unique())
all_items = sorted(train_df['item_id'].unique())
user2idx = {u:i for i,u in enumerate(all_users)}
item2idx = {it:i for i,it in enumerate(all_items)}

# Build similarity
sim_data = ItemSimilarity.build(train_df, user2idx, item2idx)

# Save similarity matrix
with open("models/item_similarity.pkl", "wb") as f:
    pickle.dump(sim_data, f)

print("Item similarity matrix saved to models/item_similarity.pkl")
