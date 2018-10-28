from flask import Blueprint, request, abort, g, jsonify
from ml_backend.users.models import User
from ml_backend import db
from ml_backend import auth

users = Blueprint("users", __name__)


@users.route("/user", methods=["POST"])
def register_user():
    json_data = request.json

    username = json_data.get("username")
    password = json_data.get("password")

    check_user_exist = User.query.filter_by(username=username).first()

    if None not in (username, password) and check_user_exist is None:
        user = User(username=username, hashed_password="")
        user.hash_password(password)
        db.session.add(user)
        db.session.commit()
        return f"Created user with username={username}", 200
    else:
        return "Either did not provide username and password or the user already exists!"


@users.route("/token", methods=["GET"])
@auth.login_required
def get_auth_token():
    token = g.user.generate_auth_token()
    return jsonify({"token": token.decode("ascii")}), 200


@auth.verify_password
def verify_password(token_or_username, password):
    user = User.verify_auth_token(token_or_username)

    if not user:
        user = User.query.filter_by(username=token_or_username).first()

        if not user or not user.verify_password(password):
            return False

    g.user = user
    return True
