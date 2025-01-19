from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
from PIL import Image
import io
import csv
import uuid
import os
import subprocess

# Variable globale pour stocker le dernier vecteur ET la dernière image brute
last_feature = None
last_image_data = None
pre = None


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
        print("loading model")

        if hasattr(model, "use_label_encoder"):
            print("okkkk")
            model.use_label_encoder = False

    with open("/app/artifacts/scaler.pickle", "rb") as f1:
        scaler = pickle.load(f1)
        print("loading scaler")

except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None

app = FastAPI()

# Mount the static files directory
app.mount("/reporting", StaticFiles(directory="/app/reporting"), name="reporting")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global last_feature, last_image_data, pre
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="Invalid file type. Please upload an image."
            )

        # On stocke l'image brute pour la sauvegarde ultérieure
        image_data = await file.read()
        last_image_data = image_data

        image = Image.open(io.BytesIO(image_data))
        features = image_to_vector(image)

        if features is None:
            raise HTTPException(status_code=500, detail="Error in image processing.")

        last_feature = features
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)

        if isinstance(prediction[0], str):
            return {"prediction": prediction[0]}
        pre = prediction[0]
        print("Predicted label (0=cat, 1=dog):", pre)
        return {"prediction": int(pre)}

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
async def validate():
    """
    L'utilisateur confirme que la prédiction précédente était correcte.
    => On enregistre l'image dans le dossier correspondant (cat/dog)
       et on stocke :
         - Dans prod_data.csv : features + colonne label
         - Dans monitor_data.csv : features + 2 colonnes (prediction=true, realite=true)
    """
    global last_feature, last_image_data, pre
    if last_feature is None or last_image_data is None:
        raise HTTPException(
            status_code=400, detail="Aucune prédiction ou image précédente."
        )

    try:
        # (1) Sauvegarde de l'image dans le dossier label (cat ou dog)
        label = "cat" if pre == 0 else "dog"
        folder_path = f"/app/data/images/{label}"
        os.makedirs(folder_path, exist_ok=True)

        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "wb") as f_img:
            f_img.write(last_image_data)

        # (2) Écriture dans prod_data.csv (features + label)
        import pandas as pd

        prod_data_path = "/app/data/prod_data.csv"
        ref_data_path = "/app/data/ref_data.csv"

        feature_array = np.array(last_feature).reshape(1, -1)
        df = pd.DataFrame(
            feature_array, columns=[f"feature{i+1}" for i in range(len(last_feature))]
        )
        df["label"] = label  # cat ou dog

        df.to_csv(prod_data_path, mode="a", header=False, index=False)
        print("Enregistré dans prod_data.csv (label = {})".format(label))

        # (3) Écriture dans monitor_data.csv (features + prediction / realite = true)
        monitor_data_path = "/app/data/monitor_data.csv"
        df_monitor = pd.DataFrame(
            feature_array, columns=[f"feature{i+1}" for i in range(len(last_feature))]
        )
        df_monitor["prediction"] = "true"
        df_monitor["realite"] = "true"

        df_monitor.to_csv(monitor_data_path, mode="a", header=False, index=False)
        print("Enregistré dans monitor_data.csv (prediction=true, realite=true).")

        # (4) Vérifier si prod_data.csv atteint 10 lignes => réentraînement éventuel
        with open(prod_data_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) >= 10:
            # Charger les nouvelles données et les données de référence
            prod_data = pd.read_csv(prod_data_path, header=None)
            ref_data = pd.read_csv(ref_data_path)
            print(prod_data.head())

            # X = toutes les colonnes sauf la dernière
            X = prod_data.iloc[:, :-1]
            # y = dernière colonne = label (cat/dog)
            y = prod_data.iloc[:, -1]

            X_scaled = scaler.transform(X)

            # Réentraîner le modèle avec un entraînement incrémental
            # model.partial_fit(X_scaled, y, classes=np.unique(y))

            with open("/app/artifacts/model.pickle", "wb") as f_model:
                pickle.dump(model, f_model)

            open(prod_data_path, "w").close()
            print("Réinitialisation de prod_data.csv après réentraînement éventuel.")

        return {
            "message": f"Image validée : label={label}, vecteur enregistré dans 2 CSV."
        }

    except Exception as e:
        print(f"Error during validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/invalidate")
async def invalidate():
    """
    L'utilisateur confirme que la prédiction précédente était incorrecte.
    => On enregistre l'image dans le "dossier inverse" :
         - si le modèle prédisait cat (0), alors c'est dog
         - si le modèle prédisait dog (1), alors c'est cat
       et on stocke :
         - Dans prod_data.csv : features + label "inverse"
         - Dans monitor_data.csv : features + (prediction=false, realite=true)
    """
    global last_feature, last_image_data, pre
    if last_feature is None or last_image_data is None:
        raise HTTPException(
            status_code=400, detail="Aucune prédiction ou image précédente."
        )

    try:
        # (1) Déterminer le label inverse
        label = "dog" if pre == 0 else "cat"
        folder_path = f"/app/data/images/{label}"
        os.makedirs(folder_path, exist_ok=True)

        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "wb") as f_img:
            f_img.write(last_image_data)

        # (2) Écriture dans prod_data.csv (features + label inverse)
        import pandas as pd

        prod_data_path = "/app/data/prod_data.csv"
        ref_data_path = "/app/data/ref_data.csv"

        feature_array = np.array(last_feature).reshape(1, -1)
        df = pd.DataFrame(
            feature_array, columns=[f"feature{i+1}" for i in range(len(last_feature))]
        )
        df["label"] = label  # cat/dog inversé

        df.to_csv(prod_data_path, mode="a", header=False, index=False)
        print("Enregistré dans prod_data.csv (label inverse = {}).".format(label))

        # (3) Écriture dans monitor_data.csv (features + prediction=false, realite=true)
        monitor_data_path = "/app/data/monitor_data.csv"
        df_monitor = pd.DataFrame(
            feature_array, columns=[f"feature{i+1}" for i in range(len(last_feature))]
        )
        df_monitor["prediction"] = "false"
        df_monitor["realite"] = "true"

        df_monitor.to_csv(monitor_data_path, mode="a", header=False, index=False)
        print("Enregistré dans monitor_data.csv (prediction=false, realite=true).")

        # (4) Vérifier si prod_data.csv atteint 10 lignes => réentraînement éventuel
        with open(prod_data_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) >= 10:
            prod_data = pd.read_csv(prod_data_path, header=None)
            ref_data = pd.read_csv(ref_data_path)
            print(prod_data.head())

            X = prod_data.iloc[:, :-1]
            y = prod_data.iloc[:, -1]

            X_scaled = scaler.transform(X)

            # model.partial_fit(X_scaled, y, classes=["cat", "dog"])  # si incrémental

            with open("/app/artifacts/model.pickle", "wb") as f_model:
                pickle.dump(model, f_model)

            open(prod_data_path, "w").close()
            print("Réinitialisation de prod_data.csv après réentraînement éventuel.")

        return {
            "message": f"Image invalidée : label inversé={label}, vecteur enregistré dans 2 CSV."
        }

    except Exception as e:
        print(f"Erreur lors de la non-validation : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report")
async def get_report():
    try:
        # Start the Streamlit app
        subprocess.Popen(["streamlit", "run", "/app/reporting/app.py", "--server.port=8501", "--server.address=0.0.0.0"])
        return {"message": "Streamlit app started on port 8501"}
    except Exception as e:
        print(f"Error starting Streamlit app: {e}")
        raise HTTPException(status_code=500, detail=str(e))