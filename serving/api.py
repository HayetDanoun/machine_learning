from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from scripts.train_model import train_model  # Assurez-vous que 'train_model' est bien importé
from scripts.train_model import load_data, preprocess_data, normalize_data
import pickle
import numpy as np
from PIL import Image
import io
from scripts.newimage import ImageTransformer
app = FastAPI()

# Fonction pour charger le modèle, le scaler et l'objet ImageTransformer
def load_model(model_path='/app/artifacts/model.pickle', scaler_path='/app/artifacts/scaler.pickle', image_transformer_path='/app/artifacts/pca.pickle'):
    """Charge le modèle, le scaler et l'objet ImageTransformer depuis les fichiers pickle."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(image_transformer_path, 'rb') as f:
        image_transformer = pickle.load(f)  # Charger l'objet ImageTransformer

    return model, scaler, image_transformer

# Route pour prédire le label de l'image
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        # Sauvegarder le fichier image temporairement
        image_path = r"C:\Users\joeto\OneDrive\Desktop\m2\machine_learning\PetImages\Cat\0.jpg"
        with open(image_path, "wb") as f:
            f.write(await file.read()) 
        # Charger le modèle, scaler, et ImageTransformer
        model, scaler, image_transformer = load_model()

        image_vector =  ImageTransformer.image_to_vector(image_path)


        # Transformer l'image en vecteur avec la méthode 'image_to_vector' de ImageTransformer
      #  image_vector = image_transformer.image_to_vector(image_path)

        if image_vector is None:
            return JSONResponse(content={"error": "Impossible de transformer l'image"}, status_code=400)

        # Normaliser le vecteur d'image
        image_vector_normalized = scaler.transform([image_vector])

        # Faire la prédiction avec le modèle
        prediction = model.predict(image_vector_normalized)

        return JSONResponse(content={"prediction": int(prediction[0])})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
