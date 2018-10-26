from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_httpauth import HTTPBasicAuth
from ml_backend.config import Config

db = SQLAlchemy()
bcrypt = Bcrypt()
auth = HTTPBasicAuth()

def create_app():
    from ml_backend.dataset.models import DataPoint
    from ml_backend.users.models import User
    from ml_backend.utils import create_db_file

    app = Flask(__name__)
    
    app.config.from_object(Config)
    db.init_app(app)
    bcrypt.init_app(app)

    create_db_file(Config().db_name, app)

    from ml_backend.dataset.routes import database
    from ml_backend.ml_serving.routes import ml_serving
    from ml_backend.users.routes import users
    app.register_blueprint(database)
    app.register_blueprint(ml_serving)
    app.register_blueprint(users)

    return app
