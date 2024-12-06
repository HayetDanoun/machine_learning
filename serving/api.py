from fastapi import FastAPI, HTTPException, File, UploadFile
import pickle
import numpy as np
from PIL import Image
import io
import csv

# Variable globale pour stocker le dernier vecteur
last_feature = None

# Fonction pour transformer l'image en vecteur
def image_to_vector(image):
    try:
        image = image.resize((32, 32)).convert("L")
        image_array = np.array(image)
        vector = image_array.flatten()
        return vector
    except Exception as e:
        print(f"Error in image processing: {e}")
        return None

# Charger le modèle et le scaler
try:
    with open("/app/artifacts/model.pickle", "rb") as f:
        model = pickle.load(f)
    
    with open("/app/artifacts/scaler.pickle", "rb") as f1:
        scaler = pickle.load(f1)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None

# Créer l'application FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global last_feature  # Accéder à la variable globale pour stocker le vecteur

    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        features = image_to_vector(image)
        
        if features is None:
            raise HTTPException(status_code=500, detail="Error in image processing.")
        
        # Sauvegarder les caractéristiques globalement
        last_feature = features

        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        
        if isinstance(prediction[0], str):
            return {"prediction": prediction[0]}
        
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
async def validate():
    global last_feature  # Accéder à la variable globale

    if last_feature is None:
        raise HTTPException(status_code=400, detail="Aucune prédiction précédente. Veuillez d'abord effectuer une prédiction.")
    
    try:
        # Sauvegarder les caractéristiques validées dans un fichier CSV
        with open("/app/data/prod_data.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(last_feature)
            print("Vecteur validé et enregistré dans le fichier CSV.")

        return {"message": "Vecteur validé et enregistré!"}

    except Exception as e:
        print(f"Error during validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
