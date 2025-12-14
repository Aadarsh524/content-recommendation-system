from fastapi import FastAPI, HTTPException
import numpy as np
import pickle
from src.serving.schemas import RecommendRequest, RecommendResponse
from src.data.load_data import load_train_val_test
from src.models.als import ALSRecommender
from src.models.baseline import train_popularity_model
from src.config import TRAINED_MODEL

app = FastAPI(title="ALS Recommendation API")

# Load artifacts

with open(f"{TRAINED_MODEL}/als_model.pkl", "rb") as f:
    als_model: ALSRecommender = pickle.load(f)

with open(f"{TRAINED_MODEL}/user_item_map.pkl", "rb") as f:
    user_item_map = pickle.load(f)

train, _, _ = load_train_val_test()
popular_items = train_popularity_model(train)


# API Endpoint

@app.get("/")
def health_check():
    return {"status": "ok", "service": "recommendation-api"}


@app.post("/recommend/als", response_model=RecommendResponse)
def recommend_als(req: RecommendRequest):
    try:
        if req.user_id not in user_item_map:
            return RecommendResponse(
                user_id=req.user_id,
                recommendations=popular_items[:req.k],
                model="ALS"
            )

        recs = als_model.recommend(
            user_id=req.user_id,
            k=req.k
        )

        return RecommendResponse(
            user_id=req.user_id,
            recommendations=recs,
            model="ALS"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
