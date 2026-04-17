"""CuraVision AI - Flask entrypoint for CDSS application."""

import os

from flask import Flask, jsonify
from sqlalchemy import inspect, text

from auth import auth_bp, bcrypt
from models import db
from routes import main_bp


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-change-this-secret-key")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///medical_system.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


db.init_app(app)
bcrypt.init_app(app)

app.register_blueprint(auth_bp, url_prefix="/auth")
app.register_blueprint(main_bp)

with app.app_context():
    db.create_all()

    # Lightweight schema upgrade for existing SQLite files used in class projects.
    inspector = inspect(db.engine)

    doctor_columns = {col["name"] for col in inspector.get_columns("doctors")}
    if "is_verified" not in doctor_columns:
        db.session.execute(text("ALTER TABLE doctors ADD COLUMN is_verified BOOLEAN NOT NULL DEFAULT 0"))
    if "verification_otp" not in doctor_columns:
        db.session.execute(text("ALTER TABLE doctors ADD COLUMN verification_otp VARCHAR(10)"))
    if "otp_expires_at" not in doctor_columns:
        db.session.execute(text("ALTER TABLE doctors ADD COLUMN otp_expires_at DATETIME"))

    patient_columns = {col["name"] for col in inspector.get_columns("patients")}
    if "generated_symptoms" not in patient_columns:
        db.session.execute(text("ALTER TABLE patients ADD COLUMN generated_symptoms TEXT NOT NULL DEFAULT ''"))
    if "photo_path" not in patient_columns:
        db.session.execute(text("ALTER TABLE patients ADD COLUMN photo_path VARCHAR(500)"))
    if "ai_insights" not in patient_columns:
        db.session.execute(text("ALTER TABLE patients ADD COLUMN ai_insights TEXT NOT NULL DEFAULT ''"))
    if "image_blob" not in patient_columns:
        db.session.execute(text("ALTER TABLE patients ADD COLUMN image_blob BLOB"))
    if "image_mime" not in patient_columns:
        db.session.execute(text("ALTER TABLE patients ADD COLUMN image_mime VARCHAR(100)"))
    if "photo_blob" not in patient_columns:
        db.session.execute(text("ALTER TABLE patients ADD COLUMN photo_blob BLOB"))
    if "photo_mime" not in patient_columns:
        db.session.execute(text("ALTER TABLE patients ADD COLUMN photo_mime VARCHAR(100)"))
    if "patient_identifier" not in patient_columns:
        db.session.execute(text("ALTER TABLE patients ADD COLUMN patient_identifier VARCHAR(64)"))

    db.session.commit()


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"success": False, "error": "File is too large. Maximum size is 16MB"}), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"success": False, "error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
