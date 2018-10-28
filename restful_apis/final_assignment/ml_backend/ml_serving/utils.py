from sklearn.tree import DecisionTreeClassifier
import numpy as np


class Model:
    def __init__(self, random_state=42):
        self.clf = DecisionTreeClassifier(random_state=random_state)
        self.eps = 1e-10

    def train(self, X, y):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        X = (X - self.mean) / (self.std + self.eps)
        self.clf.fit(X, y)
        return self.clf.score(X, y)

    def predict(self, X):
        X = (X - self.mean) / (self.std + self.eps)
        y_pred = self.clf.predict(X)[0]
        return y_pred
