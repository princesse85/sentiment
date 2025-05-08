import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# Load the model
with open('sentiment_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)

# Load the vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

st.title("Ecommerce Customer Reviews Sentiment Analysis App")


st.write("""
        Welcome to our Ecommerce Customer Reviews Analysis App. 
        This simple tool analyzes customer reviews to show you the sentiment 
        content of the review. 
        It uses machine learning to quickly highlight positive and negative
         feedback, helping you make better business decisions.""")


# Text input
user_input = st.text_area("Input The Text That You Want Analyzed Down Below: ",
                          height=100)


# Prediction button
if st.button("Predict"):
    # Transform user input to TF-IDF features
    input_features = tfidf_vectorizer.transform([user_input])
    #  Predict sentiment
    prediction = sentiment_model.predict(input_features)

    # Display the result
    st.write(f"The Text inputed is a ", prediction[0] , "Sentiment")
