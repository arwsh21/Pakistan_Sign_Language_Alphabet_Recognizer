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
        raise NotImplementedError('Subclass must implement build()')

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

class KNNRecognizer(BaseRecognizer):

    def build(self):
        return KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, weights='distance')

class SVMRecognizer(BaseRecognizer):

    def build(self):
        return SVC(C=SVM_C, kernel='rbf', probability=True, random_state=SEED)

class ForestRecognizer(BaseRecognizer):

    def build(self):
        return RandomForestClassifier(n_estimators=FOREST_TREES, random_state=SEED)

class BaselineRecognizer(BaseRecognizer):

    def build(self):
        return DummyClassifier(strategy='most_frequent')
