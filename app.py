import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------- NLTK Setup --------------------
nltk.data.path.append("/home/appuser/nltk_data")
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    model = joblib.load("spam_classifier_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -------------------- Preprocessing Tools --------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(tokens)

def predict_text(text):
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    return "Spam 🚨" if prediction == 1 else "Ham ✅"

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Spam Classifier", page_icon="📩")

st.title("📩 SMS Spam Classifier")
st.write("Enter a message below to check whether it is spam or not.")

message = st.text_area("Message")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict_text(message)
        if "Spam" in result:
            st.error(result)
        else:
            st.success(result)
