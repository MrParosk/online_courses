from flask import request, abort, jsonify, Blueprint
from ml_backend.dataset.models import DataPoint
from ml_backend import db, auth

database = Blueprint("database", __name__)


@database.route("/data/<int:id>", methods=["GET"])
@auth.login_required
def get_data(id):
    point = DataPoint.query.filter_by(id=id).first()

    if point is not None:
        return_dict = {
            "class_label": point.class_label,
            "sepal_length": point.sepal_length,
            "sepal_width": point.sepal_width,
            "pental_length": point.pental_length,
            "pental_width": point.pental_width
        }
        return jsonify(return_dict), 200
    else:
        return "The id did not exist!", 400


@database.route("/data", methods=["POST"])
@auth.login_required
def add_data():
    json_data = request.json

    class_label = json_data.get("class_label")
    sepal_length = json_data.get("sepal_length")
    sepal_width = json_data.get("sepal_width")
    pental_length = json_data.get("pental_length")
    pental_width = json_data.get("pental_width")

    if None not in (sepal_length, sepal_width, pental_length, pental_width, class_label):
        dp = DataPoint(class_label=class_label,
                       sepal_length=sepal_length,
                       sepal_width=sepal_width,
                       pental_length=pental_length,
                       pental_width=pental_width)

        db.session.add(dp)
        db.session.commit()
        return "Data sucessfully added!", 200

    else:
        return "Data had missing values!", 400
