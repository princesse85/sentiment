import streamlit as st


# Charger le modèle et le vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("🛍️ Sentiment Analysis for Ecommerce Reviews")

user_input = st.text_area("Enter your review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"
        st.success(f"Predicted sentiment: {sentiment}")
