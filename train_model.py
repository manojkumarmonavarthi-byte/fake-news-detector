import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# ---------------- ADD LABELS ----------------

fake["label"] = 0
real["label"] = 1

# ---------------- COMBINE DATA ----------------

data = pd.concat([fake, real])

# ---------------- KEEP ONLY REQUIRED COLUMNS ----------------

data = data[["text", "label"]]

# ---------------- REMOVE NULL VALUES ----------------

data.dropna(inplace=True)

# ---------------- SPLIT DATA ----------------

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- TEXT VECTORIZATION ----------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# ---------------- MODEL TRAINING ----------------

model = LogisticRegression()

model.fit(
    X_train_vectorized,
    y_train
)

# ---------------- MODEL EVALUATION ----------------

y_pred = model.predict(
    X_test_vectorized
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ---------------- SAVE MODEL ----------------

pickle.dump(
    model,
    open("model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl", "wb")
)

print("Model Trained Successfully")