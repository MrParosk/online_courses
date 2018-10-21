from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()

#def create_db_file(file_path):



def create_app():
    from ml_backend.database.models import DataPoint
    
    db_name = "test.db"

    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_name}"

    db.init_app(app)

    if not os.path.exists(db_name):
        db.create_all(app=app)

    #create_db_file(db_name)

    from ml_backend.database.routes import database
    from ml_backend.ml_serving.routes import ml_serving
    app.register_blueprint(database)
    app.register_blueprint(ml_serving)

    return app
