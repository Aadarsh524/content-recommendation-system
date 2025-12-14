from src.data.load_data import load_train_val_test
from src.models.als import ALSRecommender
from src.models.baseline import train_popularity_model
from src.evaluation.evaluate import evaluate_model
from src.config import TRAINED_MODEL
from pathlib import Path
import pickle

def build_user_items(df):
    return df.groupby("user_id")["item_id"].apply(set).to_dict()

train, _, test = load_train_val_test()

train_user_items = build_user_items(train)
test_user_items = build_user_items(test)

# Popular items for cold-start
popular_items = train_popularity_model(train)

# Train ALS
als = ALSRecommender(factors=50, iterations=20)
als.fit(train)

print("ALS matrix shape:", als.user_item_matrix.shape)


# Evaluation

def recommend_fn(user_id, seen_items, k=10):
    return als.recommend(user_id, seen_items, N=k, popular_items=popular_items)


results = evaluate_model(recommend_fn, test_user_items, train_user_items, k=10)
print(results)


processed_dir = Path(TRAINED_MODEL) 
processed_dir.mkdir(parents=True, exist_ok=True)

with open(processed_dir / "als_model.pkl", "wb") as f:
    pickle.dump(als, f)

with open(processed_dir / "user_item_map.pkl", "wb") as f:
    pickle.dump(train_user_items, f)

print("✅ ALS artifacts saved successfully")
