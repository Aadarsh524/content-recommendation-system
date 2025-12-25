from fastapi import FastAPI, HTTPException
import torch
import pickle
from src.serving.schemas import RecommendRequest, RecommendResponse
from src.data.load_data import load_train_val_test
from src.models.als import ALSRecommender
from src.models.baseline import train_popularity_model
from src.config import TRAINED_MODEL

from src.serving.schemas import SimilarRequest, SimilarResponse
from src.serving.cache import get_cache, set_cache



app = FastAPI(title="ALS Recommendation API")

# Load artifacts for ALS

with open(f"{TRAINED_MODEL}/als_model.pkl", "rb") as f:
    als_model: ALSRecommender = pickle.load(f)

with open(f"{TRAINED_MODEL}/user_item_map.pkl", "rb") as f:
    user_item_map = pickle.load(f)

train, _, _ = load_train_val_test()
popular_items = train_popularity_model(train)



# Load artifacts for NCF

with open(f"{TRAINED_MODEL}/ncf_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

ncf_model = artifacts["model"]
user2idx = artifacts["user2idx"]
item2idx = artifacts["item2idx"]
train_user_items = artifacts["train_user_items"]
n_items = artifacts["n_items"]

device = "cuda" if torch.cuda.is_available() else "cpu"
ncf_model.to(device)
ncf_model.eval()

# Popular items for cold start
train, _, _ = load_train_val_test()
popular_items = train_popularity_model(train)

# Reverse mapping (idx → item_id)
idx2item = {v: k for k, v in item2idx.items()}



# API Endpoint

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend/als", response_model=RecommendResponse)
def recommend_als(req: RecommendRequest):
    try:
        cache_key = f"als:{req.user_id}:{req.k}"
        cached = get_cache(cache_key)
        if cached:
            return cached  # already dict → FastAPI ok

        recs = als_model.recommend(
            user_id=req.user_id,
            N=req.k,
            popular_items=popular_items
        )

        response = RecommendResponse(
            user_id=req.user_id,
            recommendations=recs,
            model="ALS"
        )

        set_cache(cache_key, response.dict())
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ALS recommendation failed: {str(e)}"
        )


@app.post("/recommend/ncf", response_model=RecommendResponse)
def recommend_ncf(req: RecommendRequest):
    try:
        if req.user_id not in user2idx:
            return RecommendResponse(
                user_id=req.user_id,
                recommendations=popular_items[:req.k],
                model="NCF"
            )

        uidx = user2idx[req.user_id]
        seen_items = train_user_items.get(uidx, set())

        # Candidate items
        candidates = [
            i for i in range(n_items) if i not in seen_items
        ]

        if not candidates:
            return RecommendResponse(
                user_id=req.user_id,
                recommendations=popular_items[:req.k],
                model="NCF"
            )

        users = torch.tensor([uidx] * len(candidates), dtype=torch.long).to(device)
        items = torch.tensor(candidates, dtype=torch.long).to(device)

        with torch.no_grad():
            scores = ncf_model(users, items)

        topk_idx = torch.topk(scores, k=min(req.k, len(scores)))[1]
        rec_item_idxs = items[topk_idx].cpu().numpy()

        recommendations = [idx2item[i] for i in rec_item_idxs]

        return RecommendResponse(
            user_id=req.user_id,
            recommendations=recommendations,
            model="NCF"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"NCF recommendation failed: {str(e)}"
        )
    



@app.post("/similar", response_model=SimilarResponse)
def similar_items(req: SimilarRequest):
    try:
        items = als_model.similar_items(
            item_id=req.item_id,
            N=req.k
        )

        return SimilarResponse(
            item_id=req.item_id,
            similar_items=items
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Similar items failed: {str(e)}"
        )
