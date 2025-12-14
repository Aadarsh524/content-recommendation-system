from pydantic import BaseModel
from typing import List, Optional


class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[int]
    model: str
