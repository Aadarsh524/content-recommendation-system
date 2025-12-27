try:
    import redis
except ImportError:
    redis = None

import os
import json

REDIS_URL = os.getenv("REDIS_URL")

_client = None

def _get_client():
    global _client
    if redis is None or REDIS_URL is None:
        return None

    if _client is None:
        _client = redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def get_cache(key: str):
    client = _get_client()
    if client is None:
        return None
    value = client.get(key)
    return json.loads(value) if value else None


def set_cache(key: str, value, ttl: int = 300):
    client = _get_client()
    if client is None:
        return
    client.setex(key, ttl, json.dumps(value))
