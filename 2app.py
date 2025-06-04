import streamlit as st
from transformers import pipeline

# Charger le pipeline de sentiment avec DistilBERT
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_model()

# Titre de l'application
st.title("🧠 Analyse de Sentiment avec DistilBERT")
st.write("Entrez un texte ci-dessous pour détecter s'il est positif ou négatif.")

# Zone de texte
user_input = st.text_area("✍️ Entrez votre avis ici :", height=150)

# Prédiction
if st.button("Prédire le sentiment"):
    if not user_input.strip():
        st.warning("Veuillez entrer un texte.")
    else:
        with st.spinner("Analyse en cours..."):
            result = sentiment_analyzer(user_input)[0]
            sentiment = result['label']
            score = result['score']
            st.success(f"Sentiment prédit : **{sentiment}**")
            st.info(f"Confiance du modèle : **{score:.2%}**")
