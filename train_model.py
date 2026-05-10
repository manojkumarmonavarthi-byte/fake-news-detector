import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("news.csv")

# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    return text

df["text"] = df["text"].apply(clean_text)

# Features and labels
X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)

print("Accuracy:", score * 100)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully")