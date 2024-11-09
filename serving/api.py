import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

# Charger le modèle et le scaler
with open("/app/artifacts/model.pickle", "rb") as model_file:
    model = pickle.load(model_file)

with open("/app/artifacts/scaler.pickle", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

app = FastAPI()

# Schéma des données d'entrée
class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Convertir les features en tableau numpy pour la prédiction
        features = np.array(request.features).reshape(1, -1)

        # Normaliser les données d'entrée en utilisant le scaler chargé
        features_scaled = scaler.transform(features)

        # Faire la prédiction
        prediction = model.predict(features_scaled)
        
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
