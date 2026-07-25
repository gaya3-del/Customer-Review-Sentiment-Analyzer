# Customer Review Sentiment Analyzer

A machine-learning web application that classifies customer reviews as **Positive**, **Negative**, or **Neutral** using TF-IDF and Logistic Regression / Naive Bayes.

## Tech Stack
- Python 3.10+
- Scikit-learn
- NLTK
- Streamlit
- Pandas / NumPy

## Quick Start

```bash
# Clone the repo
git clone [github.com](https://github.com/)<your-username>/Customer-Review-Sentiment-Analyzer.git
cd Customer-Review-Sentiment-Analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (run once)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Run the app
streamlit run app/app.py
