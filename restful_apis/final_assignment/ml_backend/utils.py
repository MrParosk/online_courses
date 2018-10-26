import os
from ml_backend import db

def create_db_file(file_path, app):
    if not os.path.exists(file_path):
        db.create_all(app=app)
