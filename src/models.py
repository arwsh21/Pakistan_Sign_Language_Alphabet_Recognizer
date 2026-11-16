"""Classifier classes."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from src.config import FOREST_TREES, KNN_NEIGHBORS, SEED, SVM_C

class BaseRecognizer:

    def __init__(self):
        self.model = self.build()

    def build(self):
        # matlab khud nahi vhale ga!
        raise NotImplementedError("Subclass must implement build()")

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

class KNNRecognizer(BaseRecognizer):

  def build(self):
    return KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, weights="distance")


class SVMRecognizer(BaseRecognizer):

  def build(self):
    return SVC(C=SVM_C, kernel="rbf", probability=True, random_state=SEED)


class ForestRecognizer(BaseRecognizer):

  def build(self):
    return RandomForestClassifier(n_estimators=FOREST_TREES, random_state=SEED)


class BaselineRecognizer(BaseRecognizer):

  def build(self):
    return DummyClassifier(strategy="most_frequent")


if __name__ == "__main__":
  from sklearn.model_selection import train_test_split
  from src.features import load_dataset

  # 1. Load normalised features
  X, y = load_dataset("data/raw/psl_data.csv")

  # 2. Split train (80%) and test (20%)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=SEED, stratify=y
  )

  # 3. Create instances of all 4 models
  models = [
      ("Baseline (Dummy)", BaselineRecognizer()),
      ("KNN", KNNRecognizer()),
      ("Support Vector Machine", SVMRecognizer()),
      ("Random Forest", ForestRecognizer()),
  ]

  print("\n=== Model Comparison (Normalized Features) ===")
  for name, clf in models:
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"{name:25s}: {acc * 100:.2f}%")    