import streamlit as st
import joblib
import numpy as np

# Charger le modèle et le vectoriseur
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Titre de l'application
st.title("🧠 Prédiction de sentiment")
st.subheader("Analysez le sentiment d'un texte (positif / négatif)")

# Champ de saisie pour le texte utilisateur
user_input = st.text_area("✍️ Entrez un avis ou une phrase :")

# Lorsque l'utilisateur clique sur le bouton
if st.button("Prédire le sentiment"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer un texte pour faire une prédiction.")
    else:
        # Transformation du texte via TF-IDF
        input_transformed = vectorizer.transform([user_input])
        
        # Prédiction
        prediction = model.predict(input_transformed)

        # Affichage du résultat
        st.success(f"✅ Sentiment prédit : **{prediction[0]}**")


