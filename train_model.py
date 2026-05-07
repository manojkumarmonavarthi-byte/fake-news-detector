import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

x = data["text"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

model = LogisticRegression()

model.fit(xv_train, y_train)

pred = model.predict(xv_test)

print("Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")