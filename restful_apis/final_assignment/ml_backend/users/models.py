from flask import current_app
from ml_backend import db, bcrypt
from itsdangerous import (TimedJSONWebSignatureSerializer
                          as Serializer, BadSignature, SignatureExpired)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    username = db.Column(db.String(50), nullable=False)
    hashed_password = db.Column(db.String(50), nullable=False)

    def hash_password(self, password):
        self.hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

    def verify_password(self, input_password):
        return bcrypt.check_password_hash(self.hashed_password, input_password)

    def generate_auth_token(self, experation=600):
        s = Serializer(current_app.config["SECRET_KEY"], expires_in=experation)
        return s.dumps({"id": self.id})

    @staticmethod
    def verify_auth_token(token):
        s = Serializer(current_app.config["SECRET_KEY"])
        try:
            data = s.loads(token)
        except SignatureExpired:
            return None  # Valid token, but expired
        except BadSignature:
            return None  # Invalid token
        user = User.query.get(data["id"])
        return user
        
    def __repr__(self):
        return f"Username: {self.username}"
