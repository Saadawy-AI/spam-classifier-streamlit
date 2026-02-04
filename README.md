# Spam Email Detection using NLP

This repository contains an end-to-end **Spam Email Detection** project using **Natural Language Processing (NLP)** and **Machine Learning**, along with a **Streamlit web application** for real-time email classification.

The project demonstrates the complete workflow from data preprocessing and model training to deployment using Streamlit.

---

## 📌 Project Overview

The goal of this project is to classify emails as **Spam** or **Ham** by analyzing their textual content using NLP techniques and a machine learning model.

The solution includes:
- Data preprocessing and text cleaning
- Feature extraction using TF-IDF
- Model training with Multinomial Naive Bayes
- Model evaluation and persistence
- A Streamlit application for user interaction

---

## 🧠 NLP & Machine Learning Pipeline

1. Text normalization (lowercasing, punctuation removal)
2. Tokenization and stopword removal
3. Lemmatization
4. Feature extraction using **TF-IDF**
5. Training a **Multinomial Naive Bayes** classifier
6. Model evaluation using accuracy and classification report
7. Saving and loading the trained model for inference

---

## 🛠️ Tech Stack

- **Python**
- **Pandas & NumPy**
- **NLTK**
- **Scikit-learn**
- **TF-IDF Vectorizer**
- **Multinomial Naive Bayes**
- **Streamlit**

---

## 📂 Repository Structure

```text
.
├── Notebooke             # Project Code
├── spam_classifier_model.pkl
├── tfidf_vectorizer.pkl
├── app.py                # Streamlit application
├── requirements.txt
└── README.md


🚀 Streamlit Application
The Streamlit app allows users to:
Enter an email message
Classify the email as Spam or Ham in real time
View the prediction result instantly
To run the Streamlit app locally:
streamlit run app.py


📊 Model Performance
The model was evaluated using a validation set and achieved high accuracy with balanced precision and recall for both classes.
Detailed evaluation results can be found in the Jupyter Notebook.


🔮 Future Improvements
Experiment with different ML models (Logistic Regression, SVM)
Add deep learning models (LSTM, BERT)
Improve text preprocessing pipeline
Deploy the application on Streamlit Cloud


🤝 Feedback
Feedback and suggestions are always welcome.
---


/notebooks/
└── Spam-classifier Notebook.ipynb
