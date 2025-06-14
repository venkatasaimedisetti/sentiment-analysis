import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('data/Reviews.csv', usecols=['Text', 'Score']).dropna()

# Map scores to 3-class sentiment
def map_sentiment(score):
    if score <= 2:
        return 'negative'
    elif score == 3:
        return 'neutral'
    else:
        return 'positive'

df['Sentiment'] = df['Score'].apply(map_sentiment)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-letter characters
    return text

df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['Cleaned_Text'], df['Sentiment'], test_size=0.2, random_state=42, stratify=df['Sentiment']
)

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'models/sentiment_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
