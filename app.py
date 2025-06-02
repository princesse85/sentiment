import streamlit as st
from transformers import pipeline

# Charger le pipeline de sentiment
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_model()

# Titre de l'app
st.title("🧠 Analyse de sentiment avec DistilBERT")
st.write("Entrez un texte et obtenez le sentiment prédit (positif ou négatif).")

# Champ de texte utilisateur
user_input = st.text_area("✍️ Entrez votre texte ici :", height=150)

# Bouton de prédiction
if st.button("🔍 Prédire le sentiment"):
    if not user_input.strip():
        st.warning("Veuillez entrer un texte.")
    else:
        with st.spinner("Analyse en cours..."):
            result = sentiment_pipeline(user_input)[0]
            label = result['label']
            score = result['score']
            st.success(f"**Sentiment : {label}**\n\n🔢 Score de confiance : {score:.2f}")


