import redis
import json
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True
)

def get_cache(key):
    val = redis_client.get(key)
    return json.loads(val) if val else None

def set_cache(key, value, ttl=300):
    redis_client.setex(key, ttl, json.dumps(value))
