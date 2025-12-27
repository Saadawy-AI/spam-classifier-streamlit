import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [
        lemmatizer.lemmatize(t)
        for t in text.split()
        if t not in stop_words and len(t) > 1
    ]
    return " ".join(tokens)

def predict_text(text):
    processed = preprocess_text(text)
    vect = vectorizer.transform([processed])
    pred = model.predict(vect)[0]
    return "Spam 🚨" if pred == 1 else "Ham ✅"

st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 SMS Spam Classifier")

msg = st.text_area("Enter your message")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict_text(msg)
        if "Spam" in result:
            st.error(result)
        else:
            st.success(result)
