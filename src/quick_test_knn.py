import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('data/raw/psl_data.csv')
X = df.drop(columns=['label'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
