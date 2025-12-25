# 🎬 Content Recommendation System

Production-grade recommendation system using:
- ALS (Collaborative Filtering)
- Neural Collaborative Filtering (NCF)
- FastAPI + Docker + Redis

## Architecture
User → FastAPI → Cache → Model → Response

## Models
| Model | Use Case |
|------|---------|
| Popularity | Cold start |
| ALS | Fast collaborative filtering |
| NCF | Personalized ranking |

## API Endpoints
POST /recommend/als  
POST /recommend/ncf  
GET /health  

## Metrics
Precision@10  
Recall@10  
NDCG@10  

## Deployment
Live API: https://YOUR_URL

## Tech Stack
Python, PyTorch, implicit, FastAPI, Redis, Docker
