import streamlit as st
import requests

# URL de l'API FastAPI (en local, mais elle peut être modifiée si nécessaire)
API_URL = "http://fastapi_serving:8000/predict"  # Adaptez selon l'adresse de votre API

# Interface Streamlit
st.title("Prédiction avec le modèle")

# Demander à l'utilisateur d'entrer les caractéristiques
st.write("Entrez les valeurs des caractéristiques pour la prédiction:")

# Créer des champs de saisie pour chaque caractéristique
features = []
for i in range(5):  # Supposons que vous ayez 5 caractéristiques, ajustez si nécessaire
    value = st.number_input(f"Caractéristique {i + 1}", value=0.0)
    features.append(value)

# Bouton pour soumettre la requête
if st.button("Prédire"):
    # Préparer les données pour l'API
    data = {"features": features}
    
    # Envoyer la requête à l'API de prédiction
    response = requests.post(API_URL, json=data)
    
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.write(f"La prédiction obtenue est : {prediction}")
    else:
        st.error("Erreur dans la prédiction. Veuillez réessayer.")
