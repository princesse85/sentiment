import streamlit as st
import joblib  

# ✅ Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_sentiment(text):
    # Transform the input text using the TF-IDF vectorizer
    X_input = vectorizer.transform([text])
    prediction = model.predict(X_input)[0]
    return prediction

# 🎨 Streamlit App
st.title("🛍 E-commerce Sentiment Analysis")
st.write("Enter a product review and get the predicted sentiment!")

# 📝 Text input
user_input = st.text_area("🗣 Enter a review:")

# 🔍 Prediction button
if st.button("Analyze Sentiment"):
    if user_input:
        prediction = predict_sentiment(user_input)
        if prediction == 1 or prediction == "Positive":
            st.success("✅ Sentiment: Positive 😊")
        else:
            st.error("❌ Sentiment: Negative 😠")
    else:
        st.warning("⚠ Please enter some text.")
        
