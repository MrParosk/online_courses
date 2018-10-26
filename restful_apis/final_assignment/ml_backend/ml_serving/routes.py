from flask import Blueprint, request, abort
import numpy as np
import pickle
from ml_backend.dataset.utils import query_data
from ml_backend import auth
from ml_backend.ml_serving.utils import Model
from ml_backend.config import Config

ml_serving = Blueprint("ml_serving", __name__)
model_path = f"ml_backend/static/{Config().model_name}"


@ml_serving.route("/train", methods=["GET"])
@auth.login_required
def train_model():
    data = query_data()

    X = [[d["sepal_length"], d["sepal_width"], d["pental_length"], d["pental_width"]] for d in data]
    X = np.array(X)

    y = [[d["class_label"]] for d in data]
    y = np.array(y)

    model = Model()
    score = model.train(X, y)
    pickle.dump(model, open(model_path, 'wb'))

    return f"Training accuracy: {str(score)}", 200


@ml_serving.route("/prediction", methods=["POST"])
@auth.login_required
def predict():
    json_data = request.json

    sepal_length = json_data.get("sepal_length")
    sepal_width = json_data.get("sepal_width")
    pental_length = json_data.get("pental_length")
    pental_width = json_data.get("pental_width")

    if None not in (sepal_length, sepal_width, pental_length, pental_width):
        X = [sepal_length, sepal_width, pental_length, pental_width]
        X = np.array(X).reshape(1,4)

        model = pickle.load(open(model_path, "rb"))
        y_pred = model.predict(X)

        return f"Predicted class label: {str(y_pred)}", 200
    else:
        return "Data had missing values!", 400
