"""
app.py
Streamlit web app for the Customer Review Sentiment Analyzer.
Loads the trained model + vectorizer from models/ and serves live predictions.

Run with:
    streamlit run app.py
"""

import pickle
import streamlit as st
from utils import predict_sentiment

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


@st.cache_resource
def load_artifacts():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None


st.set_page_config(page_title="Customer Review Sentiment Analyzer", page_icon="🗣️")
st.title("🗣️ Customer Review Sentiment Analyzer")
st.write("Enter a customer review below to classify its sentiment.")

model, vectorizer = load_artifacts()

if model is None:
    st.error(
        "Model files not found in `models/`. Run the training notebook "
        "(`notebooks/sentiment_analysis.ipynb`) first — it saves "
        "`sentiment_model.pkl` and `tfidf_vectorizer.pkl` there."
    )
else:
    user_input = st.text_area(
        "Review text:", height=150,
        placeholder="e.g. The product arrived late and stopped working after two days..."
    )

    if st.button("Analyze Sentiment"):
        if not user_input.strip():
            st.warning("Please enter some review text first.")
        else:
            try:
                result = predict_sentiment(user_input, model, vectorizer)
                emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}.get(result['sentiment'], "🤔")
                st.subheader(f"{emoji} Predicted Sentiment: **{result['sentiment'].upper()}**")
                st.write(f"Confidence: {result['confidence'] * 100:.2f}%")
            except Exception as e:
                st.error(f"Something went wrong while predicting: {e}")

st.markdown("---")
st.caption("TF-IDF + Logistic Regression")
