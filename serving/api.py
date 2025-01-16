from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
from PIL import Image
import io
import csv
import os

# Variable globale pour stocker le dernier vecteur
last_feature = None
pre = None

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
        print(" loading model ")

        if hasattr(model, 'use_label_encoder'):
            print("okkkk")
            model.use_label_encoder = False

    with open("/app/artifacts/scaler.pickle", "rb") as f1:
        scaler = pickle.load(f1)
        print(" loading scaler ")

except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None

# Créer l'application FastAPI
app = FastAPI()

# Mount the static files directory
app.mount("/reporting", StaticFiles(directory="/app/reporting"), name="reporting")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global last_feature  # Accéder à la variable globale pour stocker le vecteur
    global pre
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
        pre = prediction[0]
        print(pre)
        return {"prediction": int(prediction[0])}
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
async def validate():
    global last_feature  # Accéder à la variable globale
    global pre
    if last_feature is None:
        raise HTTPException(status_code=400, detail="Aucune prédiction précédente. Veuillez d'abord effectuer une prédiction.")
    
    try:
        # Sauvegarder les caractéristiques validées dans un fichier CSV
        prod_data_path = "/app/data/prod_data.csv"
        ref_data_path = "/app/data/ref_data.csv"
        import pandas as pd

        with open(prod_data_path, "a", newline="") as f:
            feature_array = np.array(last_feature).reshape(1, -1)
            df = pd.DataFrame(feature_array, columns=[f"feature{i+1}" for i in range(len(last_feature))])
        
            df['prediction'] = "true"
            df['realite'] = "true"

            df.to_csv(prod_data_path, mode='a', header=False, index=False)

            print("Vecteur validé et enregistré dans le fichier CSV.")

        # Vérifier si prod_data.csv atteint 10 lignes
        with open(prod_data_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if len(rows) >= 10:
            # Charger les nouvelles données et les données de référence
            prod_data = pd.read_csv(prod_data_path, header=None)
            ref_data = pd.read_csv(ref_data_path)
            print(prod_data.head())

            # Préparation des données pour l'entraînement
            X = prod_data.iloc[:, :-1]  # Sélectionne toutes les colonnes sauf la dernière
            y = prod_data.iloc[:, -1]

            # Normalisation des données
            X_scaled = scaler.transform(X)

            # Réentraîner le modèle avec un entraînement incrémental
            # model.partial_fit(X_scaled, y, classes=np.unique(y))

            # Sauvegarder le modèle et le scaler
            with open("/app/artifacts/model.pickle", "wb") as f:
                pickle.dump(model, f)
          
            # Réinitialiser prod_data.csv pour les prochaines entrées
            open(prod_data_path, "w").close()
            print("Modèle réentraîné incrémentalement et prod_data.csv réinitialisé.")

        return {"message": "Vecteur validé et enregistré!"}

    except Exception as e:
        print(f"Error during validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/invalidate")
async def invalidate():
    global last_feature  # Accéder à la variable globale
    global pre
    if last_feature is None:
        raise HTTPException(status_code=400, detail="Aucune prédiction précédente. Veuillez d'abord effectuer une prédiction.")
    
    try:
        # Sauvegarder les caractéristiques invalidées dans un fichier CSV
        prod_data_path = "/app/data/prod_data.csv"
        ref_data_path = "/app/data/ref_data.csv"
        import pandas as pd

        # Sauvegarder les données dans le fichier CSV
        with open(prod_data_path, "a", newline="") as f:
            feature_array = np.array(last_feature).reshape(1, -1)
            df = pd.DataFrame(feature_array, columns=[f"feature{i+1}" for i in range(len(last_feature))])
            
            df['prediction'] = "false"
            df['realite'] = "true"
            df.to_csv(prod_data_path, mode='a', header=False, index=False)

            print("Vecteur non validé et enregistré avec l'étiquette inversée dans le fichier CSV.")

        # Vérifier si prod_data.csv atteint 10 lignes
        with open(prod_data_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) >= 10:
            # Charger les nouvelles données et les données de référence
            prod_data = pd.read_csv(prod_data_path, header=None)
            ref_data = pd.read_csv(ref_data_path)
            print(prod_data.head())

            # Préparation des données pour l'entraînement
            X = prod_data.iloc[:, :-1]  # Sélectionne toutes les colonnes sauf la dernière
            y = prod_data.iloc[:, -1]

            # Normalisation des données
            X_scaled = scaler.transform(X)

            # Réentraîner le modèle avec un entraînement incrémental
            # model.partial_fit(X_scaled, y, classes=np.unique(y))

            # Sauvegarder le modèle et le scaler
            with open("/app/artifacts/model.pickle", "wb") as f:
                pickle.dump(model, f)

            # Réinitialiser prod_data.csv pour les prochaines entrées
            open(prod_data_path, "w").close()
            print("Modèle réentraîné incrémentalement et prod_data.csv réinitialisé.")

        return {"message": "Vecteur non validé et enregistré avec l'étiquette inversée!"}

    except Exception as e:
        print(f"Erreur lors de la non-validation : {e}")
        raise HTTPException(status_code=500, detail=str(e))