from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.api.recommender_service import RecommenderService

# Initialize app and recommender
app = FastAPI(title="Recommendation API", version="1.0")
recommender = RecommenderService(device="cpu")  # change to "cuda" if GPU available

# ----- Pydantic request models -----
class UserRequest(BaseModel):
    user_id: int
    top_k: Optional[int] = None
    k: Optional[int] = None

    def get_k(self) -> int:
        if self.k is not None:
            return self.k
        if self.top_k is not None:
            return self.top_k
        return 10

class ItemRequest(BaseModel):
    item_id: int
    top_k: Optional[int] = None
    k: Optional[int] = None

    def get_k(self) -> int:
        if self.k is not None:
            return self.k
        if self.top_k is not None:
            return self.top_k
        return 10

# ----- Health check -----
@app.get("/health")
def health():
    return {"status": "ok", "message": "API is running"}

# ----- ALS recommendation endpoint -----
@app.post("/recommend")
def recommend(request: UserRequest):
    if request.user_id not in recommender.user2idx:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        k = request.get_k()
        items = recommender.recommend(request.user_id, k)
        return {"user_id": request.user_id, "recommendations": items}
    except ValueError as e:
        # our explicit validation error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----- NCF recommendation endpoint -----
@app.post("/recommend_neural")
def recommend_neural(request: UserRequest):
    if request.user_id not in recommender.user2idx:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        k = request.get_k()
        items = recommender.recommend_neural(request.user_id, k)
        return {"user_id": request.user_id, "recommendations": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----- Item similarity endpoint -----
@app.post("/similar")
def similar(request: ItemRequest):
    if request.item_id not in recommender.item2idx:
        raise HTTPException(status_code=404, detail="Item not found")
    try:
        k = request.get_k()
        items = recommender.similar(request.item_id, k)
        return {"item_id": request.item_id, "similar_items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))