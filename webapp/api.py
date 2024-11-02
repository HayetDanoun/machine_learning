import streamlit as st
import requests

# Titre de l'application
st.title("Application de Duplication de Mot")

# Saisie du mot par l'utilisateur
user_input = st.text_input("Entrez un mot:")

if st.button("Envoyer"):
    if user_input:
        # Appel à l'API
        url = "http://serving-api:8080/duplicate"
        response = requests.post(url, json={"word": user_input})

        if response.status_code == 200:
            duplicated_word = response.json().get("duplicated_word")
            st.success("Mot dupliqué: {}".format(duplicated_word))
        else:
            st.error("Erreur lors de la connexion à l'API.")
    else:
        st.warning("Veuillez entrer un mot.")
