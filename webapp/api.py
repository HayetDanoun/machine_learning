import streamlit as st
import requests
from PIL import Image
import io

# ------------------------
# Configuration de la page
# ------------------------
st.set_page_config(
    page_title="Image Classification Demo",
    layout="wide",  # 'centered' ou 'wide'
    initial_sidebar_state="expanded"
)

# ------------------------
# Feuille de style CSS personnalisée
# ------------------------
page_bg = """
<style>
/* Fond général de la page */
body {
    background-color: #f0f2f6;
}

div.stButton {
    display: flex;
    justify-content: center; /* Centre horizontalement */
    align-items: center; /* Centre verticalement */
    margin: 20px 0; /* Espacement autour des boutons */
}


/* Titres et sous-titres */
h1, h2, h3, h4 {
    color: #333333; /* Gris foncé professionnel */
    font-family: 'Roboto', Arial, sans-serif; /* Police moderne et professionnelle */
    text-align: center; /* Centré pour un meilleur impact visuel */
    background-color: #ecf0f1; /* Légère couleur d'arrière-plan */
    border-radius: 10px;
}

/* Bouton */
/* Bouton par défaut */
div.stButton > button:first-child {
    background-color: #007BFF; /* Bleu standard */
    color: white; /* Texte en blanc */
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-size: 1rem;
    font-weight: bold;
    transition: background-color 0.3s, color 0.3s; /* Animation douce */
}

/* Bouton au survol */
div.stButton > button:first-child:hover {
    background-color: #0056b3; /* Bleu plus foncé */
    color: white;
}

/* Bouton lors du clic (active) */
div.stButton > button:first-child:active {
    background-color: #003f7f; /* Encore plus foncé */
    color: white; /* Texte devient blanc */
}



/* Couleur des boîtes de messages (info, error, success) */
.element-container .stAlert {
    border-radius: 10px;
}

/* Champs de sélection */
.css-1cifp57, .css-19ih76x, .stSelectbox label {
    color: #333333;
}



</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ------------------------
# Constantes
# ------------------------
API_PREDICT_URL = "http://fastapi_serving:8000/predict"
API_VALIDATE_URL = "http://fastapi_serving:8000/validate"
API_INVALIDATE_URL = "http://fastapi_serving:8000/invalidate"


# ========================
# ========================
with st.sidebar:
    st.title("🐾 Classification d'Images")
    st.markdown(
        """
        Cette application **Streamlit** permet de prédire 
        si l'image chargée représente un **chat** ou un **chien**.
        
        **Étapes** :
        1. Téléchargez votre image.
        2. Obtenez la prédiction.
        3. Validez ou corrigez la prédiction.
        """
    )
    st.markdown("---")
    st.markdown("**API Endpoints :**")
    st.markdown(f"- Prediction : `{API_PREDICT_URL}`")
    st.markdown(f"- Validate : `{API_VALIDATE_URL}`")
    st.markdown(f"- Invalidate : `{API_INVALIDATE_URL}`")


# ------------------------
# En-tête / Bandeau
# ------------------------
st.title("🐾 Démonstration de classification d'images 🐾")
st.markdown("""

---

Bienvenue dans notre application de classification d'images ! Cette démonstration utilise un modèle d'intelligence artificielle pour prédire si l'image téléchargée correspond à un chat ou à un chien.

Ce mini-projet permet de :

    Téléchargez une image (JPG/PNG).
    Obtenez la prédiction du modèle.
    Fournissez un feedback pour améliorer les performances.

---

""")

# ------------------------
# Chargement de l'image
# ------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Chargement de l'image :arrow_down:")

    uploaded_file = st.file_uploader(
        label="Téléchargez une image (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Afficher l'image
        image = Image.open(uploaded_file)
        st.image(image, caption="Aperçu de l'image chargée", use_column_width=True)

with col2:
    st.subheader("2. Prédiction :crystal_ball:")
    
    # Ajouter un espacement entre le titre et le bouton
    st.write("")  

    # Espace dédié à la prédiction
    prediction_label_placeholder = st.empty()

    # Vérification que l'image est chargée avant d'afficher le bouton
    if st.button("Obtenir la prédiction"):
        if uploaded_file is not None:
            # Convertir l'image en bytes pour l'envoyer à l'API
            img_bytes = io.BytesIO()
            # On force le format JPEG de sortie
            image.save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()
            files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}

            try:
                response = requests.post(API_PREDICT_URL, files=files)

                if response.status_code == 200:
                    prediction = response.json().get("prediction", None)
                    if prediction == 0:
                        prediction_label = "Chat (Miaouuuu 🐱)"
                    elif prediction == 1:
                        prediction_label = "Chien (Wouffff 🐶)"
                    else:
                        prediction_label = "Désolé, nous ne connaissons pas cet animal."

                    prediction_label_placeholder.info(f"**Prédiction du modèle** : {prediction_label}")

                else:
                    st.error(f"Erreur lors de la prédiction: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Erreur lors de l'appel à l'API: {str(e)}")
        else:
            st.warning("Veuillez d'abord télécharger une image avant de demander une prédiction.")

# ------------------------
# Zone de Feedback
# ------------------------
st.markdown("---")
st.subheader("3. Feedback :white_check_mark:")
# Texte centré verticalement et horizontalement
st.markdown("""
<div style="
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    height: 100px; /* Ajustez cette valeur pour définir la hauteur du conteneur */
    margin: 1rem 0;
    background-color: #ecf0f1; /* Légère couleur d'arrière-plan */
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
">
    <p style="
        font-size: 1.2rem;
        font-family: 'Roboto', Arial, sans-serif;
        color: #333333;
    ">
        Si la prédiction vous semble correcte, validez-la !<br>
        Sinon, indiquez la bonne étiquette pour aider à l'amélioration du modèle.
    </p>
</div>
""", unsafe_allow_html=True)


feedback_col1, feedback_col2 = st.columns(2)

with feedback_col1:
    st.subheader("Valider la prédiction (si correcte)")

    if st.button("Valider"):
        try:
            validate_response = requests.post(API_VALIDATE_URL)
            if validate_response.status_code == 200:
                st.success("La prédiction a été validée et le feedback enregistré.")
                st.balloons()
            else:
                st.error(f"Erreur lors de la validation : {validate_response.text}")
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API de validation: {str(e)}")

with feedback_col2:
    st.subheader("Corriger la prédiction (si incorrecte)")

    # Sélection de l'étiquette correcte si la prédiction est fausse
    correct_label = st.selectbox(
        "Sélectionnez l'étiquette correcte :",
        ["", "0 (Chat)", "1 (Chien)"],
        index=0
    )

    if st.button("Envoyer feedback"):
        if correct_label == "":
            st.warning("Veuillez sélectionner une étiquette correcte avant de soumettre.")
        else:
            try:
                # En envoyant l'étiquette correcte (ex: "0" ou "1") à l'API
                payload = {"correct_label": correct_label.split(" ")[0]}  # récupère juste le 0 ou 1
                invalidate_response = requests.post(API_INVALIDATE_URL, json=payload)
                if invalidate_response.status_code == 200:
                    st.success("Feedback envoyé avec l'étiquette correcte. Merci !")
                    st.snow()
                else:
                    st.error(f"Erreur lors de la non-validation : {invalidate_response.text}")
            except Exception as e:
                st.error(f"Erreur lors de l'appel à l'API de non-validation: {str(e)}")

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.caption("✨ Application Streamlit de démonstration — M2 Data Science / Machine Learning ✨")

