import streamlit as st
import requests
from PIL import Image
import io

# URL de l'API FastAPI
API_URL = "http://fastapi_serving:8000/predict"
VALIDATE_URL = "http://fastapi_serving:8000/validate"
INVALIDATE_URL = "http://fastapi_serving:8000/invalidate"

st.title("Prédiction avec le modèle d'image")

# Télécharger une image
uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Convertir l'image téléchargée en un format attendu
    image = Image.open(uploaded_image)
    image = image.resize((224, 224))  # Redimensionne si nécessaire pour le modèle

    # Afficher l'image téléchargée
    st.image(image, caption="Image téléchargée", use_column_width=True)

    if st.button("Prédire"):
        # Convertir l'image en bytes et l'envoyer à l'API
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        # Sending the image with the correct format
        files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
        
        try:
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                prediction = response.json().get("prediction", "Aucune prédiction reçue.")
                if prediction == 0:
                      st.write("La prédiction obtenue est : cat")
                elif prediction == 1:
                       st.write("La prédiction obtenue est : dog")   
            else:
                st.error(f"Erreur lors de la prédiction: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API: {str(e)}")

    # Bouton de validation
    if st.button("Valider"):
        try:
            validate_response = requests.post(VALIDATE_URL)

            if validate_response.status_code == 200:
                st.success("Vecteur validé et enregistré!")
            else:
                st.error(f"Erreur lors de la validation du vecteur: {validate_response.text}")
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API de validation: {str(e)}")

    # Bouton de non-validation
    if st.button("Non Valider"):
        try:
            invalidate_response = requests.post(INVALIDATE_URL)

            if invalidate_response.status_code == 200:
                st.success("Vecteur non validé et enregistré avec l'étiquette inversée!")
            else:
                st.error(f"Erreur lors de la non-validation du vecteur: {invalidate_response.text}")
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API de non-validation: {str(e)}")
