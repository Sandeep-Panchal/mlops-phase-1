from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

class Request(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Sentiment API running"}

@app.post("/predict")
def get_prediction(req: Request):
    result = predict(req.text)
    return {"Sentiment": result}