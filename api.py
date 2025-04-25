from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

from src.inference import RecommendationEngine

app = FastAPI()

# Load the model
recommender = RecommendationEngine()

class UserQuery(BaseModel):
    user_id: str
    channel: str
    device_type: str
    query_text: str
    time: datetime

@app.post("/recommend/")
async def get_recommendations(query: UserQuery):
    recommendations = recommender.get_recommendations(query.model_dump())
    return {"recommendations": recommendations}
