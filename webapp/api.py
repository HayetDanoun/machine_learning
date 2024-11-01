import streamlit as st
import requests
import json

# Configuration de l'URL de l'API de serving
API_URL = "http://serving-api:8080/predict"

def main():
    st.title("Interface de Prédiction")

    # Uploader un fichier (image/vidéo/audio)
    uploaded_file = st.file_uploader("Choisissez un fichier", type=["jpg", "jpeg", "png", "mp4", "wav"])

    if uploaded_file is not None:
        # Afficher le fichier uploadé
        st.image(uploaded_file, caption='Fichier uploadé', use_column_width=True)

        # Bouton pour faire la prédiction
        if st.button("Faire une prédiction"):
            # Préparation des données pour l'appel à l'API
            files = {'file': uploaded_file.getvalue()}  # Pour les fichiers

            try:
                # Envoyer une requête POST à l'API
                response = requests.post(API_URL, files=files)
                response.raise_for_status()  # Vérifier si la requête a réussi

                # Afficher le résultat
                prediction = response.json()
                st.success(f"Prédiction : {prediction['message']}")
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de l'appel à l'API : {e}")

if __name__ == "__main__":
    main()
