from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_httpauth import HTTPBasicAuth
import os
from ml_backend.config import Config

db = SQLAlchemy()
bcrypt = Bcrypt()
auth = HTTPBasicAuth()

#def create_db_file(file_path):

def create_app():
    app = Flask(__name__)

    from ml_backend.dataset.models import DataPoint
    from ml_backend.users.models import User
    
    app.config.from_object(Config)
    db.init_app(app)
    bcrypt.init_app(app)

    if not os.path.exists(Config().db_name):
        db.create_all(app=app)

    #create_db_file(db_name)

    from ml_backend.dataset.routes import database
    from ml_backend.ml_serving.routes import ml_serving
    from ml_backend.users.routes import users
    app.register_blueprint(database)
    app.register_blueprint(ml_serving)
    app.register_blueprint(users)

    return app
