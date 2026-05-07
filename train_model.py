import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

news = pd.concat([fake, true], axis=0)

news = news[["text", "label"]]

news.dropna(inplace=True)

x = news["text"]
y = news["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

x = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(x_train, y_train)

pred = model.predict(x_test)

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Saved Successfully")