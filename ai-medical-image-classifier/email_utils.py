"""Email notification helper for OTP and scan updates.

Supports direct provider API integration for faster setup:
- Resend (MAIL_PROVIDER=resend)
- SendGrid (MAIL_PROVIDER=sendgrid)
- SMTP fallback (MAIL_PROVIDER=smtp or unset)
"""

import os
import smtplib
from email.message import EmailMessage

import requests

from models import Doctor


def _send_via_smtp(recipients, subject, body, sender):
    """Send mail through SMTP (default legacy path)."""
    mail_username = os.getenv("MAIL_USERNAME")
    mail_password = os.getenv("MAIL_PASSWORD")
    mail_server = os.getenv("MAIL_SERVER", "smtp.gmail.com")
    mail_port = int(os.getenv("MAIL_PORT", "587"))
    mail_use_tls = os.getenv("MAIL_USE_TLS", "true").lower() == "true"

    if not mail_username or not mail_password:
        print("MAIL_USERNAME/MAIL_PASSWORD not configured. SMTP email skipped.")
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(body)

    try:
        with smtplib.SMTP(mail_server, mail_port, timeout=20) as smtp:
            if mail_use_tls:
                smtp.starttls()
            smtp.login(mail_username, mail_password)
            smtp.send_message(message)
        return True
    except Exception as exc:
        print(f"Failed to send SMTP email: {exc}")
        return False


def _send_via_resend(recipients, subject, body, sender):
    """Send mail through Resend API."""
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        print("RESEND_API_KEY not configured. Resend email skipped.")
        return False

    payload = {
        "from": sender,
        "to": recipients,
        "subject": subject,
        "text": body,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post("https://api.resend.com/emails", json=payload, headers=headers, timeout=20)
        if 200 <= response.status_code < 300:
            return True
        print(f"Resend API failed ({response.status_code}): {response.text}")
        return False
    except Exception as exc:
        print(f"Failed to send Resend email: {exc}")
        return False


def _send_via_sendgrid(recipients, subject, body, sender):
    """Send mail through SendGrid API."""
    api_key = os.getenv("SENDGRID_API_KEY")
    if not api_key:
        print("SENDGRID_API_KEY not configured. SendGrid email skipped.")
        return False

    payload = {
        "personalizations": [{"to": [{"email": email} for email in recipients]}],
        "from": {"email": sender},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.sendgrid.com/v3/mail/send", json=payload, headers=headers, timeout=20
        )
        if 200 <= response.status_code < 300:
            return True
        print(f"SendGrid API failed ({response.status_code}): {response.text}")
        return False
    except Exception as exc:
        print(f"Failed to send SendGrid email: {exc}")
        return False


def _send_email(recipients, subject, body):
    """Dispatch email through configured provider."""
    provider = (os.getenv("MAIL_PROVIDER", "smtp") or "smtp").strip().lower()
    sender = os.getenv("MAIL_SENDER") or os.getenv("MAIL_USERNAME") or "noreply@example.com"

    if provider == "resend":
        return _send_via_resend(recipients=recipients, subject=subject, body=body, sender=sender)
    if provider == "sendgrid":
        return _send_via_sendgrid(recipients=recipients, subject=subject, body=body, sender=sender)

    return _send_via_smtp(recipients=recipients, subject=subject, body=body, sender=sender)


def send_new_scan_notification(patient_name, prediction, confidence, timestamp):
    """Send a notification email to all registered doctors.

    Uses configured provider from MAIL_PROVIDER.
    Returns True if email dispatch was attempted successfully, else False.
    """
    recipients = [doctor.email for doctor in Doctor.query.all() if doctor.email]
    if not recipients:
        return True

    subject = "New Medical Scan Uploaded"
    body = (
        "A new scan has been uploaded.\n"
        f"Patient: {patient_name}\n"
        f"Result: {prediction} ({confidence:.2f}%)\n"
        f"Time: {timestamp}\n"
    )

    return _send_email(recipients=recipients, subject=subject, body=body)


def send_verification_otp(doctor_email, doctor_name, otp_code):
    """Send account verification OTP to a newly registered doctor."""
    subject = "CuraVision AI Email Verification OTP"
    body = (
        f"Hello Dr. {doctor_name},\n\n"
        "Your CuraVision AI verification OTP is:\n"
        f"{otp_code}\n\n"
        "This OTP expires in 10 minutes.\n"
        "If you did not request this account, please ignore this email."
    )
    return _send_email(recipients=[doctor_email], subject=subject, body=body)
