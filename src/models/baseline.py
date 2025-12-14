def train_popularity_model(train_df):
    """
    Returns items sorted by popularity (descending)
    """
    return (
        train_df.groupby("item_id")
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )



def recommend_popular(popular_items, seen_items, k=10):
    recommendations = []
    for item in popular_items:
        if item  not in seen_items:
            recommendations.append(item)
        if len(recommendations) == k:
            break
    return recommendations