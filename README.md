# 📩 Spam Message Classifier

An end-to-end **NLP + Machine Learning** project that classifies text messages as **Spam** or **Ham**, with a **Streamlit** web app for real-time predictions.

The pipeline covers everything from text preprocessing and feature extraction to model training, evaluation, and deployment.

---

## 📌 Overview

The goal is to detect spam messages by analyzing their textual content using NLP techniques and a trained classification model.

**Pipeline:**
- Text cleaning and normalization
- Tokenization, stopword removal, and lemmatization
- Feature extraction with **TF-IDF**
- Classification using **Multinomial Naive Bayes**
- Model evaluation (accuracy, precision, recall, F1-score)
- Interactive prediction via a Streamlit interface

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data | Pandas, NumPy |
| NLP | NLTK |
| ML | Scikit-learn (TF-IDF, Multinomial Naive Bayes) |
| App | Streamlit |
| Serialization | Joblib |

---

## 📂 Project Structure

```
.
├── .devcontainer/              # Dev container config
├── Notebook/                   # EDA, preprocessing, training & evaluation
├── app.py                      # Streamlit application
├── spam_classifier_model.pkl   # Trained model
├── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python runtime version
└── README.md
```

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/Saadawy-AI/spam-classifier-streamlit.git
cd spam-classifier-streamlit
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, type a message, and click **Predict** to see whether it's classified as Spam or Ham.

---

## 📊 Model Performance

The model was evaluated on a held-out validation set, achieving high accuracy with balanced precision and recall across both classes. Full evaluation details (confusion matrix, classification report) are available in the [Notebook](./Notebook).

---

## 🔮 Future Improvements

- [ ] Compare against other models (Logistic Regression, SVM)
- [ ] Experiment with deep learning approaches (LSTM, BERT)
- [ ] Show prediction confidence/probability in the UI, not just the label
- [ ] Deploy publicly on Streamlit Community Cloud and link the live demo here
- [ ] Add unit tests for the preprocessing pipeline

---

## 👤 Author

**Mohamed Saadawy**
Data Engineering & ML Engineering Student

- GitHub: [@Saadawy-AI](https://github.com/Saadawy-AI)
- LinkedIn: [muhammad-saadawy](https://linkedin.com/in/muhammad-saadawy)

---

## 📝 License

This project is open for learning and reference purposes. Feel free to fork and build on it.
