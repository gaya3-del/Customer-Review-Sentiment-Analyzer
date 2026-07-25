"""
utils.py
Shared preprocessing and prediction functions for the Customer Review
Sentiment Analyzer. Imported by both the training notebook and app.py
so the exact same cleaning logic is used at train time and inference time.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Make sure required NLTK data is available (safe to call every import)
for pkg, path in [('stopwords', 'corpora/stopwords'),
                   ('wordnet', 'corpora/wordnet'),
                   ('omw-1.4', 'corpora/omw-1.4')]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Label mapping — change this in ONE place if you move to a 3-class dataset later.
LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/punctuation/numbers, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)              # HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)      # URLs
    text = re.sub(r'[^a-z\s]', ' ', text)            # punctuation/digits/special chars
    text = re.sub(r'\s+', ' ', text).strip()         # collapse whitespace
    return text


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize."""
    text = clean_text(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def predict_sentiment(text: str, model, vectorizer) -> dict:
    """Run the full inference pipeline on one piece of raw text."""
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return {
        'text': text,
        'sentiment': LABEL_MAP_INV[pred],
        'confidence': round(float(max(proba)), 4)
    }
