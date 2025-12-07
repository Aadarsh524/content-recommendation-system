from pydantic import BaseModel

class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10

class SimilarRequest(BaseModel):
    item_id: int
    k: int = 10

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[int]

class SimilarResponse(BaseModel):
    item_id: int
    similar_items: list[int]
