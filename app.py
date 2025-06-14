import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# Streamlit app
st.title("Sentiment Analysis")
text_input = st.text_area("Enter a review:", "")

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_input = clean_text(text_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]
        st.success(f"Predicted Sentiment: **{prediction.upper()}**")
