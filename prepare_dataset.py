import pandas as pd

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# Combine
df = pd.concat([fake, true])

# Keep needed columns
df = df[["text", "label"]]

# Save
df.to_csv("news.csv", index=False)

print("news.csv created successfully")