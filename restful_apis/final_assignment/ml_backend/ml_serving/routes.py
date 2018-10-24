from flask import Blueprint, request, abort
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle
from ml_backend.dataset.routes import query_data
import numpy as np

ml_serving = Blueprint("ml_serving", __name__)
model_filename = "ml_backend/static/model.pickle"

@ml_serving.route("/train", methods=["GET"])
def train_model():
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


@ml_serving.route("/prediction", methods=["POST"])
def predict():
    json_data = request.json

    sepal_length = json_data.get("sepal_length")
    sepal_width = json_data.get("sepal_width")
    pental_length = json_data.get("pental_length")
    pental_width = json_data.get("pental_width")

    if None not in (sepal_length, sepal_width, pental_length, pental_width):
        X = [sepal_length, sepal_width, pental_length, pental_width]
        X = np.array(X).reshape(1,4)

        with open(model_filename, "rb") as file:
            clf = pickle.load(file)

        y_pred = clf.predict(X)

        return str(y_pred)
    else:
        return abort(400)
