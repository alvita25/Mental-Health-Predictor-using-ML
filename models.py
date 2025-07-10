from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime

db = SQLAlchemy()
bcrypt = Bcrypt()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    email = db.Column(db.String(150), unique=True)
    password_hash = db.Column(db.String(200))
    predictions = db.relationship('Prediction', backref='user', lazy=True)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result_xgb = db.Column(db.String(100))
    certainty_xgb = db.Column(db.Float)

    result_lgbm = db.Column(db.String(100))
    certainty_lgbm = db.Column(db.Float)

    result_logistic = db.Column(db.String(100))
    certainty_logistic = db.Column(db.Float)

    result_nb = db.Column(db.String(100))
    certainty_nb = db.Column(db.Float)

    result_rf = db.Column(db.String(100))
    certainty_rf = db.Column(db.Float)

    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Code block to create the database and tables when run directly
if __name__ == '__main__':
    from flask import Flask

    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    bcrypt.init_app(app)

    with app.app_context():
        db.create_all()
        print("✅ Database and tables created successfully.")
