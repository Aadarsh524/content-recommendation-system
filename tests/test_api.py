import pytest
from fastapi.testclient import TestClient
from src.serving.api import app

client = TestClient(app)

# ---------- HEALTH ----------

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# ---------- ALS ----------

def test_recommend_als_known_user():
    res = client.post(
        "/recommend/als",
        json={"user_id": 40, "k": 5}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "ALS"
    assert len(body["recommendations"]) == 5


def test_recommend_als_unknown_user():
    res = client.post(
        "/recommend/als",
        json={"user_id": 999999, "k": 5}
    )
    assert res.status_code == 200
    assert len(res.json()["recommendations"]) == 5


# ---------- NCF ----------

def test_recommend_ncf_known_user():
    res = client.post(
        "/recommend/ncf",
        json={"user_id": 40, "k": 5}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "NCF"
    assert len(body["recommendations"]) == 5


def test_recommend_ncf_unknown_user():
    res = client.post(
        "/recommend/ncf",
        json={"user_id": 999999, "k": 5}
    )
    assert res.status_code == 200
    assert len(res.json()["recommendations"]) == 5
