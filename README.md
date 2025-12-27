# Content Recommendation System (ALS + NCF)

A production-ready recommendation system using **Collaborative Filtering (ALS)** and **Neural Collaborative Filtering (NCF)**, exposed via a **FastAPI REST service**, cached with **Redis**, containerized using **Docker**, and deployed on **Render**.

## 🔗 Live API

Base URL:  
https://content-recommendation-system-xsd2.onrender.com

### Health Check
GET /health

### User Recommendations (ALS)
POST /recommend/als

### User Recommendations (NCF)
POST /recommend/ncf

### Item-to-Item Similarity
POST /similar

## 🏗 Architecture

Client (curl / frontend)
        |
        v
FastAPI (Uvicorn)
        |
        +-- ALS Model (implicit)
        |
        +-- NCF Model (PyTorch)
        |
        +-- Redis Cache
        |
        v
Pre-trained Model Artifacts (Pickle)



## ✨ Features

- Hybrid recommender system (ALS + NCF)
- User-based recommendations
- Item-to-item similarity search
- Cold-start handling using popularity baseline
- Redis caching for low-latency inference
- REST API with FastAPI
- Dockerized for portability
- Deployed on Render (cloud-ready)


## API Usage Examples 
curl -X POST https://content-recommendation-system-xsd2.onrender.com/recommend/als \
-H "Content-Type: application/json" \
-d '{"user_id":40,"k":10}'


## 🚀 Run Locally

```bash
git clone https://github.com/your-username/content-recommendation-system.git
cd content-recommendation-system
pip install -r requirements.txt
uvicorn src.serving.api:app --reload
