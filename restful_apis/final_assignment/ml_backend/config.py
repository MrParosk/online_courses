class Config:
    db_name = "test.db"
    model_name = "model.pickle"

    SECRET_KEY = "derp"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_name}"
