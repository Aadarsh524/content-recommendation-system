import redis
from fastapi import Depends
from src.api.recommender_service import RecommenderService

redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

def get_cache():
    return redis_client

_service = RecommenderService()

def get_service():
    return _service
