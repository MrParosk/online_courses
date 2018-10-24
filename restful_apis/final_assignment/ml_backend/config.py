class Config:
    db_name = "test.db"

    SECRET_KEY = "derp"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_name}"
