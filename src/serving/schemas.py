from pydantic import BaseModel
from typing import List

class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[int]
    model: str

class SimilarRequest(BaseModel):
    item_id: int
    k: int = 10

class SimilarResponse(BaseModel):
    item_id: int
    similar_items: List[int]
