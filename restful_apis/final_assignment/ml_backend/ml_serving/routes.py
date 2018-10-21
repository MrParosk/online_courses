from flask import Blueprint
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle
from ml_backend.database.routes import query_data

ml_serving = Blueprint("ml_serving", __name__)


@ml_serving.route("/train_model", methods=["GET"])
def train_model():
    model_filename = "ml_backend/static/model.pickle"

    data = query_data()

    X = [[d["sepal_length"], d["sepal_width"], d["pental_length"], d["pental_width"]] for d in data]
    X = np.array(X)

    y = [[d["class_label"]] for d in data]
    y = np.array(y)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    score = clf.score(X, y)

    
    pickle.dump(clf, open(model_filename, 'wb'))
    return str(score)
