"""Main clinical routes for dashboard, uploads, case views, and AI helper."""

from datetime import datetime
import io
import os
import re

import numpy as np
from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, send_file, send_from_directory, session, url_for
from PIL import Image
import torch
from tensorflow.keras.models import load_model
from transformers import CLIPModel, CLIPProcessor
from werkzeug.utils import secure_filename

from ai_logic import ai_helper_assessment, build_ai_insights_summary, generate_prediction_symptoms
from auth import login_required
from email_utils import send_new_scan_notification
from models import Patient, db


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

main_bp = Blueprint("main", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PNEUMONIA_MODEL_PATH = os.path.join(BASE_DIR, "models", "model.h5")
UPLOAD_ROOT = os.path.join(BASE_DIR, "uploads")
XRAY_UPLOAD_DIR = os.path.join(UPLOAD_ROOT, "xrays")
PHOTO_UPLOAD_DIR = os.path.join(UPLOAD_ROOT, "photos")

os.makedirs(XRAY_UPLOAD_DIR, exist_ok=True)
os.makedirs(PHOTO_UPLOAD_DIR, exist_ok=True)

CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}
CLASS_DESCRIPTIONS = {
    "NORMAL": "The X-ray appears normal with no major signs of pneumonia.",
    "PNEUMONIA": "The X-ray shows signs consistent with pneumonia.",
}
PNEUMONIA_THRESHOLD = float(os.getenv("PNEUMONIA_THRESHOLD", "0.40"))

if not os.path.exists(PNEUMONIA_MODEL_PATH):
    raise RuntimeError(f"Pneumonia model not found at {PNEUMONIA_MODEL_PATH}")

pneumonia_model = load_model(PNEUMONIA_MODEL_PATH)

try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
except Exception:
    clip_model = None
    clip_processor = None


ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp"}
PATIENT_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,32}$")


def allowed_file(filename):
    """Validate file extension from filename."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def normalize_patient_identifier(value):
    """Normalize and validate patient ID used for patient portal access."""
    candidate = (value or "").strip().upper()
    if not candidate:
        return None, "Patient ID is required."
    if not PATIENT_IDENTIFIER_PATTERN.fullmatch(candidate):
        return None, "Patient ID must be 3-32 characters (letters, numbers, - or _)."
    return candidate, None


def preprocess_image(img_file):
    """Preprocess image for CNN model."""
    img = Image.open(img_file.stream)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def detect_chest_xray(img_array):
    """Check if uploaded image looks like a chest X-ray using CLIP."""
    if clip_model is None or clip_processor is None:
        return True, 1.0

    img_uint8 = (img_array[0] * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    candidate_labels = [
        "a chest x-ray radiograph",
        "a photograph or screenshot",
        "a hand or foot or knee or spine x-ray",
    ]

    inputs = clip_processor(text=candidate_labels, images=pil_img, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = torch.softmax(outputs.logits_per_image[0], dim=0).numpy()

    chest_xray_prob = float(probs[0])
    return chest_xray_prob > 0.5, chest_xray_prob


def predict_pneumonia(img_array):
    """Predict NORMAL/PNEUMONIA with configurable threshold."""
    raw_probability = float(pneumonia_model.predict(img_array, verbose=0)[0][0])
    class_idx = 1 if raw_probability >= PNEUMONIA_THRESHOLD else 0

    confidence = raw_probability if class_idx == 1 else (1 - raw_probability)
    return CLASS_NAMES[class_idx], confidence, raw_probability


def validate_patient_form(form_data):
    """Validate mandatory patient fields."""
    name = (form_data.get("patient_name") or "").strip()
    age_text = (form_data.get("age") or "").strip()
    symptoms = (form_data.get("symptoms") or "").strip()

    if not name:
        return None, None, None, "Patient name is required."

    if not age_text.isdigit():
        return None, None, None, "Age must be a valid number."

    age = int(age_text)
    if age < 0 or age > 120:
        return None, None, None, "Age must be between 0 and 120."

    if not symptoms:
        return None, None, None, "Symptoms are required."

    return name, age, symptoms, None


def save_upload(file_obj, target_dir):
    """Save uploaded file and return relative path."""
    safe_filename = secure_filename(file_obj.filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    stored_filename = f"{timestamp}_{safe_filename}"
    absolute_path = os.path.join(target_dir, stored_filename)
    file_obj.stream.seek(0)
    file_obj.save(absolute_path)

    relative_path = os.path.relpath(absolute_path, UPLOAD_ROOT)
    return relative_path.replace("\\", "/")


def delete_upload(relative_path):
    """Delete a file under uploads safely when case data is removed."""
    if not relative_path:
        return

    normalized = relative_path.replace("\\", "/").strip("/")
    absolute_path = os.path.abspath(os.path.join(UPLOAD_ROOT, normalized))
    upload_root_abs = os.path.abspath(UPLOAD_ROOT)

    # Prevent accidental deletion outside uploads/.
    if not absolute_path.startswith(upload_root_abs):
        return

    if os.path.isfile(absolute_path):
        os.remove(absolute_path)


@main_bp.route("/")
def index():
    """Root route sends user to login or dashboard."""
    if session.get("doctor_id"):
        return redirect(url_for("main.dashboard"))
    return redirect(url_for("auth.login"))


@main_bp.route("/dashboard")
@login_required
def dashboard():
    """Advanced dashboard with stats, high-risk cases, notifications, and recent table."""
    total_scans = Patient.query.count()
    pneumonia_cases = Patient.query.filter_by(prediction="PNEUMONIA").count()
    normal_cases = Patient.query.filter_by(prediction="NORMAL").count()

    recent_cases = Patient.query.order_by(Patient.created_at.desc()).limit(12).all()
    notifications = Patient.query.order_by(Patient.created_at.desc()).limit(5).all()
    high_risk_cases = (
        Patient.query.filter(Patient.prediction == "PNEUMONIA", Patient.confidence >= 85)
        .order_by(Patient.created_at.desc())
        .limit(8)
        .all()
    )

    return render_template(
        "dashboard.html",
        total_scans=total_scans,
        pneumonia_cases=pneumonia_cases,
        normal_cases=normal_cases,
        recent_cases=recent_cases,
        notifications=notifications,
        high_risk_cases=high_risk_cases,
    )


@main_bp.route("/upload")
@login_required
def upload_page():
    """Enhanced upload form page."""
    return render_template("upload.html")


@main_bp.route("/predict", methods=["POST"])
@login_required
def predict_route():
    """Process X-ray upload, run model, save case, and send notifications."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Chest X-ray image is required."}), 400

    xray_file = request.files["file"]
    patient_photo = request.files.get("patient_photo")

    if xray_file.filename == "" or not allowed_file(xray_file.filename):
        return jsonify({"success": False, "error": "Valid chest X-ray image is required."}), 400

    if patient_photo and patient_photo.filename and not allowed_file(patient_photo.filename):
        return jsonify({"success": False, "error": "Patient photo format is invalid."}), 400

    patient_name, patient_age, manual_symptoms, error_message = validate_patient_form(request.form)
    if error_message:
        return jsonify({"success": False, "error": error_message}), 400

    patient_identifier, patient_id_error = normalize_patient_identifier(request.form.get("patient_identifier"))
    if patient_id_error:
        return jsonify({"success": False, "error": patient_id_error}), 400

    try:
        xray_file.stream.seek(0)
        img_array = preprocess_image(xray_file)
        is_chest_xray, detector_confidence = detect_chest_xray(img_array)

        if not is_chest_xray:
            return jsonify({
                "success": False,
                "error": "Uploaded image is not a valid chest X-ray.",
                "detector_confidence": round(detector_confidence * 100, 2),
            }), 400

        prediction, confidence, raw_probability = predict_pneumonia(img_array)
        confidence_percent = round(confidence * 100, 2)

        xray_file.stream.seek(0)
        xray_blob = xray_file.read()
        xray_file.stream.seek(0)
        xray_mime = xray_file.mimetype or "application/octet-stream"

        generated_symptoms = generate_prediction_symptoms(prediction)
        ai_summary = build_ai_insights_summary(
            prediction=prediction,
            confidence=confidence_percent,
            manual_symptoms=manual_symptoms,
            generated_symptoms=generated_symptoms,
        )

        image_path = save_upload(xray_file, XRAY_UPLOAD_DIR)
        photo_path = None
        photo_blob = None
        photo_mime = None
        if patient_photo and patient_photo.filename:
            patient_photo.stream.seek(0)
            photo_blob = patient_photo.read()
            patient_photo.stream.seek(0)
            photo_mime = patient_photo.mimetype or "application/octet-stream"
            photo_path = save_upload(patient_photo, PHOTO_UPLOAD_DIR)

        created_at = datetime.utcnow()
        patient = Patient(
            patient_identifier=patient_identifier,
            name=patient_name,
            age=patient_age,
            symptoms=manual_symptoms,
            generated_symptoms=generated_symptoms,
            image_path=image_path,
            photo_path=photo_path,
            image_blob=xray_blob,
            image_mime=xray_mime,
            photo_blob=photo_blob,
            photo_mime=photo_mime,
            prediction=prediction,
            confidence=confidence_percent,
            ai_insights=ai_summary,
            created_at=created_at,
        )

        db.session.add(patient)
        db.session.commit()

        send_new_scan_notification(
            patient_name=patient_name,
            prediction=prediction,
            confidence=confidence_percent,
            timestamp=created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        return jsonify({
            "success": True,
            "patient_id": patient.id,
            "patient_identifier": patient.patient_identifier,
            "patient_name": patient_name,
            "prediction": prediction,
            "confidence": confidence_percent,
            "description": CLASS_DESCRIPTIONS[prediction],
            "message": "Consult a medical professional" if prediction == "PNEUMONIA" else "No major symptoms detected",
            "manual_symptoms": manual_symptoms,
            "generated_symptoms": generated_symptoms,
            "ai_insights": ai_summary,
            "timestamp": created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "pneumonia_probability": round(raw_probability * 100, 2),
            "pneumonia_threshold": round(PNEUMONIA_THRESHOLD * 100, 2),
        })
    except Exception as exc:
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500


@main_bp.route("/cases")
@login_required
def cases_list():
    """Cases table page."""
    cases = Patient.query.order_by(Patient.created_at.desc()).all()
    return render_template("cases.html", cases=cases)


@main_bp.route("/cases/<int:case_id>")
@login_required
def case_detail(case_id):
    """Detailed case view with images, symptoms, prediction and insights."""
    case = Patient.query.get_or_404(case_id)
    return render_template("case_detail.html", case=case)


@main_bp.route("/patient/login", methods=["GET", "POST"])
def patient_login():
    """Patient access page that accepts only patient ID."""
    if request.method == "POST":
        patient_identifier, error = normalize_patient_identifier(request.form.get("patient_identifier"))
        if error:
            flash(error, "danger")
            return render_template("patient_login.html", patient_identifier="")
        return redirect(url_for("main.patient_reports", patient_identifier=patient_identifier))

    return render_template("patient_login.html", patient_identifier="")


@main_bp.route("/patient/reports/<string:patient_identifier>")
def patient_reports(patient_identifier):
    """Show all reports mapped to the provided patient ID."""
    normalized_id, error = normalize_patient_identifier(patient_identifier)
    if error:
        flash("Invalid Patient ID.", "danger")
        return redirect(url_for("main.patient_login"))

    reports = Patient.query.filter_by(patient_identifier=normalized_id).order_by(Patient.created_at.desc()).all()
    return render_template("patient_reports.html", patient_identifier=normalized_id, reports=reports)


@main_bp.route("/patient/reports/<string:patient_identifier>/<int:case_id>")
def patient_report_detail(patient_identifier, case_id):
    """Show one report if it belongs to the provided patient ID."""
    normalized_id, error = normalize_patient_identifier(patient_identifier)
    if error:
        flash("Invalid Patient ID.", "danger")
        return redirect(url_for("main.patient_login"))

    report = Patient.query.filter_by(id=case_id, patient_identifier=normalized_id).first_or_404()
    return render_template("patient_case_detail.html", case=report, patient_identifier=normalized_id)


@main_bp.route("/cases/<int:case_id>/delete", methods=["POST"])
@login_required
def delete_case(case_id):
    """Delete a case record and related uploaded files."""
    case = Patient.query.get_or_404(case_id)
    case_name = case.name

    try:
        delete_upload(case.image_path)
        delete_upload(case.photo_path)
        db.session.delete(case)
        db.session.commit()
        flash(f"Case data for {case_name} was deleted.", "success")
        return redirect(url_for("main.cases_list"))
    except Exception:
        db.session.rollback()
        flash("Failed to delete case data. Please try again.", "danger")
        return redirect(url_for("main.case_detail", case_id=case_id))


@main_bp.route("/cases/<int:case_id>/image/<string:kind>")
@login_required
def case_image(case_id, kind):
    """Serve case images from database blob first, then filesystem fallback."""
    case = Patient.query.get_or_404(case_id)

    if kind == "xray":
        if case.image_blob:
            return send_file(io.BytesIO(case.image_blob), mimetype=case.image_mime or "image/jpeg")
        if case.image_path:
            return send_from_directory(UPLOAD_ROOT, case.image_path)
        abort(404)

    if kind == "photo":
        if case.photo_blob:
            return send_file(io.BytesIO(case.photo_blob), mimetype=case.photo_mime or "image/jpeg")
        if case.photo_path:
            return send_from_directory(UPLOAD_ROOT, case.photo_path)
        abort(404)

    abort(404)


@main_bp.route("/patient/reports/<string:patient_identifier>/<int:case_id>/image/<string:kind>")
def patient_report_image(patient_identifier, case_id, kind):
    """Serve report images for patient access when patient ID matches."""
    normalized_id, error = normalize_patient_identifier(patient_identifier)
    if error:
        abort(404)

    case = Patient.query.filter_by(id=case_id, patient_identifier=normalized_id).first_or_404()

    if kind == "xray":
        if case.image_blob:
            return send_file(io.BytesIO(case.image_blob), mimetype=case.image_mime or "image/jpeg")
        if case.image_path:
            return send_from_directory(UPLOAD_ROOT, case.image_path)
        abort(404)

    if kind == "photo":
        if case.photo_blob:
            return send_file(io.BytesIO(case.photo_blob), mimetype=case.photo_mime or "image/jpeg")
        if case.photo_path:
            return send_from_directory(UPLOAD_ROOT, case.photo_path)
        abort(404)

    abort(404)


@main_bp.route("/ai-helper", methods=["GET", "POST"])
@login_required
def ai_helper_page():
    """CDSS AI helper page using rule-based logic."""
    result = None
    form_values = {
        "patient_name": "",
        "age": "",
        "symptoms": "",
        "prediction": "",
    }

    if request.method == "POST":
        form_values = {
            "patient_name": (request.form.get("patient_name") or "").strip(),
            "age": (request.form.get("age") or "").strip(),
            "symptoms": (request.form.get("symptoms") or "").strip(),
            "prediction": (request.form.get("prediction") or "").strip(),
        }

        age_value = form_values["age"] if form_values["age"].isdigit() else "0"

        result = ai_helper_assessment(
            patient_name=form_values["patient_name"] or "Patient",
            age=int(age_value),
            symptoms=form_values["symptoms"],
            prediction_result=form_values["prediction"],
        )

    return render_template("ai_helper.html", result=result, form_values=form_values)


@main_bp.route("/about")
def about():
    """Health check endpoint with model metadata."""
    return jsonify(
        {
            "name": "CuraVision AI Clinical Decision Support System",
            "classes": CLASS_NAMES,
            "pneumonia_threshold_percent": round(PNEUMONIA_THRESHOLD * 100, 2),
        }
    )


@main_bp.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    """Serve uploaded X-ray and patient photo files for authenticated users."""
    return send_from_directory(UPLOAD_ROOT, filename)
