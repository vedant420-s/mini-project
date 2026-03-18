"""Authentication and email verification routes for doctors."""

from datetime import datetime, timedelta
from functools import wraps
import random

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from flask_bcrypt import Bcrypt

from email_utils import send_verification_otp
from models import Doctor, db


auth_bp = Blueprint("auth", __name__)
bcrypt = Bcrypt()


def login_required(view_function):
    """Protect routes so only authenticated doctors can access them."""

    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if "doctor_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("auth.login"))
        return view_function(*args, **kwargs)

    return wrapped_view


def generate_otp_code():
    """Generate a 6-digit OTP string."""
    return f"{random.randint(100000, 999999)}"


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """Register doctor and send verification OTP email."""
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters long.", "danger")
            return render_template("register.html")

        existing = Doctor.query.filter_by(email=email).first()
        if existing:
            flash("A doctor account with this email already exists.", "danger")
            return render_template("register.html")

        otp_code = generate_otp_code()
        doctor = Doctor(
            name=name,
            email=email,
            password_hash=bcrypt.generate_password_hash(password).decode("utf-8"),
            is_verified=False,
            verification_otp=otp_code,
            otp_expires_at=datetime.utcnow() + timedelta(minutes=10),
        )

        db.session.add(doctor)
        db.session.commit()

        otp_sent = send_verification_otp(doctor_email=email, doctor_name=name, otp_code=otp_code)

        if otp_sent:
            flash("Registration successful. OTP sent to your email.", "success")
        else:
            flash("Registration successful, but OTP email could not be sent. Check mail settings.", "warning")
            if current_app.debug:
                flash(f"Development OTP: {otp_code}", "info")

        return redirect(url_for("auth.verify_email", email=email))

    return render_template("register.html")


@auth_bp.route("/verify-email", methods=["GET", "POST"])
def verify_email():
    """Verify doctor email with OTP code."""
    email = (request.args.get("email") or request.form.get("email") or "").strip().lower()
    doctor = Doctor.query.filter_by(email=email).first() if email else None

    if request.method == "POST":
        otp_code = (request.form.get("otp") or "").strip()

        if doctor is None:
            flash("Doctor account not found for verification.", "danger")
            return render_template("verify_email.html", email=email)

        if doctor.is_verified:
            flash("Email already verified. Please log in.", "info")
            return redirect(url_for("auth.login"))

        if not otp_code:
            flash("OTP is required.", "danger")
            return render_template("verify_email.html", email=email)

        if not doctor.verification_otp or doctor.verification_otp != otp_code:
            flash("Invalid OTP code.", "danger")
            return render_template("verify_email.html", email=email)

        if doctor.otp_expires_at is None or doctor.otp_expires_at < datetime.utcnow():
            flash("OTP expired. Please resend OTP.", "warning")
            return render_template("verify_email.html", email=email)

        doctor.is_verified = True
        doctor.verification_otp = None
        doctor.otp_expires_at = None
        db.session.commit()

        flash("Email verified successfully. Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("verify_email.html", email=email)


@auth_bp.route("/resend-otp", methods=["POST"])
def resend_otp():
    """Resend OTP email for unverified doctor account."""
    email = (request.form.get("email") or "").strip().lower()
    doctor = Doctor.query.filter_by(email=email).first()

    if doctor is None:
        flash("Account not found.", "danger")
        return redirect(url_for("auth.verify_email", email=email))

    if doctor.is_verified:
        flash("Account already verified. Please log in.", "info")
        return redirect(url_for("auth.login"))

    doctor.verification_otp = generate_otp_code()
    doctor.otp_expires_at = datetime.utcnow() + timedelta(minutes=10)
    db.session.commit()

    otp_sent = send_verification_otp(
        doctor_email=doctor.email,
        doctor_name=doctor.name,
        otp_code=doctor.verification_otp,
    )

    if otp_sent:
        flash("A new OTP has been sent to your email.", "success")
    else:
        flash("Could not send OTP email. Check mail settings.", "warning")
        if current_app.debug:
            flash(f"Development OTP: {doctor.verification_otp}", "info")

    return redirect(url_for("auth.verify_email", email=email))


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Authenticate verified doctors and start a session."""
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template("login.html")

        doctor = Doctor.query.filter_by(email=email).first()
        if doctor and bcrypt.check_password_hash(doctor.password_hash, password):
            if not doctor.is_verified:
                flash("Please verify your email before login.", "warning")
                return redirect(url_for("auth.verify_email", email=doctor.email))

            session.clear()
            session["doctor_id"] = doctor.id
            session["doctor_name"] = doctor.name
            session["doctor_email"] = doctor.email
            flash("Login successful.", "success")
            return redirect(url_for("main.dashboard"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")


@auth_bp.route("/logout")
def logout():
    """End doctor session."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
