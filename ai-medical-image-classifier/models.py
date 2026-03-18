"""Database models for doctors and patients."""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class Doctor(db.Model):
    """Stores doctor accounts used for authentication and notifications."""

    __tablename__ = "doctors"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    is_verified = db.Column(db.Boolean, nullable=False, default=False)
    verification_otp = db.Column(db.String(10), nullable=True)
    otp_expires_at = db.Column(db.DateTime, nullable=True)


class Patient(db.Model):
    """Stores uploaded scan metadata and prediction output."""

    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    generated_symptoms = db.Column(db.Text, nullable=False, default="")
    image_path = db.Column(db.String(500), nullable=False)
    photo_path = db.Column(db.String(500), nullable=True)
    image_blob = db.Column(db.LargeBinary, nullable=True)
    image_mime = db.Column(db.String(100), nullable=True)
    photo_blob = db.Column(db.LargeBinary, nullable=True)
    photo_mime = db.Column(db.String(100), nullable=True)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    ai_insights = db.Column(db.Text, nullable=False, default="")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
