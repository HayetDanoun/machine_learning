import os
import io
import requests
import streamlit as st
from PIL import Image

# --- Config page
st.set_page_config(page_title="Image Classification Demo", layout="wide", initial_sidebar_state="expanded")

# --- Base URL de l'API (à définir dans Render > Environment Variables du service webapp)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_PREDICT_URL = f"{API_BASE_URL}/predict"
API_VALIDATE_URL = f"{API_BASE_URL}/validate"
API_INVALIDATE_URL = f"{API_BASE_URL}/invalidate"

# --- CSS (raccourci)
st.markdown("""
<style>
div.stButton { display:flex; justify-content:center; align-items:center; margin:20px 0; }
h1, h2, h3, h4 { color:#333; font-family: 'Roboto', Arial, sans-serif; text-align:center; background:#ecf0f1; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🐾 Classification d'Images")
    st.markdown("**API Endpoints :**")
    st.code(API_PREDICT_URL)
    st.code(API_VALIDATE_URL)
    st.code(API_INVALIDATE_URL)

st.title("🐾 Démonstration de classification d'images 🐾")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Chargement de l'image ⬇️")
    uploaded_file = st.file_uploader("Téléchargez une image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Aperçu de l'image chargée", use_container_width=True)

with col2:
    st.subheader("2. Prédiction 🔮")
    prediction_label_placeholder = st.empty()

    if st.button("Obtenir la prédiction"):
        if uploaded_file is None:
            st.warning("Veuillez d'abord télécharger une image.")
        else:
            # Recréer les bytes (uploaded_file peut être déjà lu ailleurs)
            img_buf = io.BytesIO()
            image.save(img_buf, format="JPEG")
            img_buf.seek(0)
            files = {"file": ("image.jpg", img_buf, "image/jpeg")}
            try:
                resp = requests.post(API_PREDICT_URL, files=files, timeout=30)
                if resp.ok:
                    pred = resp.json().get("prediction")
                    if pred == 0:
                        label = "Chat (Miaouuuu 🐱)"
                    elif pred == 1:
                        label = "Chien (Wouffff 🐶)"
                    else:
                        label = f"Inconnu ({pred})"
                    prediction_label_placeholder.info(f"**Prédiction du modèle** : {label}")
                else:
                    st.error(f"Erreur {resp.status_code} : {resp.text}")
            except Exception as e:
                st.error(f"Erreur lors de l'appel à l'API : {e}")

st.markdown("---")
st.subheader("3. Feedback ✅")

fb1, fb2 = st.columns(2)

with fb1:
    st.subheader("Valider la prédiction (si correcte)")
    if st.button("Valider"):
        try:
            r = requests.post(API_VALIDATE_URL, timeout=15)
            st.success("Prédiction validée. Merci !") if r.ok else st.error(r.text)
        except Exception as e:
            st.error(f"Erreur de validation : {e}")

with fb2:
    st.subheader("Corriger la prédiction (si incorrecte)")
    correct_label = st.selectbox("Sélectionnez l'étiquette correcte :", ["", "0 (Chat)", "1 (Chien)"], index=0)
    if st.button("Envoyer feedback"):
        if not correct_label:
            st.warning("Sélectionnez une étiquette.")
        else:
            payload = {"correct_label": correct_label.split(" ")[0]}
            try:
                r = requests.post(API_INVALIDATE_URL, json=payload, timeout=15)
                st.success("Feedback envoyé. Merci !") if r.ok else st.error(r.text)
            except Exception as e:
                st.error(f"Erreur de feedback : {e}")

st.markdown("---")
st.caption("✨ Application Streamlit de démonstration — M2 Data Science / Machine Learning ✨")
