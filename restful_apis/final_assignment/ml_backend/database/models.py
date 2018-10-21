from ml_backend import db

class DataPoint(db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    class_label = db.Column(db.Integer, nullable=False)

    sepal_length = db.Column(db.Float, nullable=False)
    sepal_width = db.Column(db.Float, nullable=False)
    pental_length = db.Column(db.Float, nullable=False)
    pental_width = db.Column(db.Float, nullable=False)
