from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI()

# Load model
stacked_model = joblib.load("stacked_model.pkl")

class MicrobiomeData(BaseModel):
    PC1: float = Field(..., example=0.1)
    PC2: float = Field(..., example=0.2)
    PC3: float = Field(..., example=0.3)
    Shannon_Index: float = Field(..., example=1.5)
    Richness: float = Field(..., example=25)
    Evenness: float = Field(..., example=0.8)

@app.post("/predict/")
async def predict(data: MicrobiomeData):
    try:
        input_data = pd.DataFrame([data.dict()])
        prediction = stacked_model.predict(input_data)
        probability = stacked_model.predict_proba(input_data)[:, 1]
        return {"prediction": prediction.tolist(), "probability": probability.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Gut Microbiome API is live!"}
