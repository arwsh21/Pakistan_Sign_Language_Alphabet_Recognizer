import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Load your recorded CSV
df = pd.read_csv("data/raw/psl_data.csv")

# 2. X is all 63 coordinates, y is the sign name
X = df.drop(columns=["label"])
y = df["label"]

# 3. Split 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train the K-Nearest Neighbors model
clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf.fit(X_train, y_train)

# 5. Test on the unseen 20%
accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")