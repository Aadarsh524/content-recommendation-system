
from src.evaluation.evaluate import evaluate_model
from src.data.load_data import load_train_val_test
from src.models.baseline import  train_popularity_model, recommend_popular

def build_user_items(df):
    return (
        df.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )


train, _, test = load_train_val_test()

popular_items = train_popularity_model(train)

train_user_items = build_user_items(train)
test_user_items = build_user_items(test)


def recommend_fn(user_id, seen_items, k=10):
    return recommend_popular(popular_items, seen_items, k)

results = evaluate_model(
    recommend_fn,
    test_user_items,
    train_user_items,
    k=10
)

print(results)