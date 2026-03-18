# AI-Powered Clinical Decision Support System (CDSS)

Educational Flask application for chest X-ray triage, doctor workflow, and symptom-based clinical support.

## Medical Disclaimer

This project is strictly for learning and academic demonstration.

- Not approved for real clinical diagnosis
- Not a replacement for doctors or radiologists
- Always confirm with qualified medical professionals

## What This App Does

- Doctor registration and login
- Email OTP verification before doctor login access
- Patient case creation with X-ray upload and optional patient photo
- Two-stage AI flow:
  - Chest X-ray suitability detection (CLIP)
  - Pneumonia probability prediction (TensorFlow model)
- Auto-generated symptom hints and case insight text
- Case history and case details view
- AI helper page for symptom/risk recommendation support

## Tech Stack

- Backend: Flask, Flask-SQLAlchemy, Flask-Bcrypt
- Database: SQLite
- AI: TensorFlow/Keras + CLIP (Transformers)
- Email: Resend or SendGrid direct API (SMTP fallback)
- UI: Jinja templates + Tailwind-based SaaS layout

## Project Structure (Core Files)

```text
ai-medical-image-classifier/
  app.py
  auth.py
  routes.py
  models.py
  ai_logic.py
  email_utils.py
  test_app.py
  requirements.txt
  templates/
    base_app.html
    login.html
    register.html
    verify_email.html
    dashboard.html
    upload.html
    cases.html
    case_detail.html
    ai_helper.html
  models/
    model.h5
    xray_detector.h5
```

## Environment Variables

Set these in your shell before running the app.

```powershell
$env:SECRET_KEY="your-secret-key"
$env:MAIL_PROVIDER="resend"   # resend | sendgrid | smtp
$env:MAIL_SENDER="onboarding@yourdomain.com"
$env:RESEND_API_KEY="re_xxxxx"
$env:SENDGRID_API_KEY="SG.xxxxx"
$env:MAIL_USERNAME="your-email@gmail.com"
$env:MAIL_PASSWORD="your-app-password"
$env:PNEUMONIA_THRESHOLD="0.40"
```

Notes:

- `MAIL_PROVIDER` defaults to `smtp` if not set.
- For `resend`, set `RESEND_API_KEY` and use a verified `MAIL_SENDER`.
- For `sendgrid`, set `SENDGRID_API_KEY` and use a verified `MAIL_SENDER`.
- For `smtp`, set `MAIL_USERNAME` and `MAIL_PASSWORD`.
- `PNEUMONIA_THRESHOLD` defaults to `0.40` if not set.

## Setup

### 1. Create and activate virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Ensure models are present

Required model files:

- `models/model.h5`
- `models/xray_detector.h5`

### 4. Run application

```powershell
python app.py
```

Open:

- http://127.0.0.1:5000

## Main Routes

Auth:

- `/auth/register`
- `/auth/verify-email`
- `/auth/resend-otp`
- `/auth/login`
- `/auth/logout`

App:

- `/dashboard`
- `/upload`
- `/predict` (POST)
- `/cases`
- `/cases/<case_id>`
- `/ai-helper`
- `/about`

## Testing

Run basic health and pipeline tests:

```powershell
python test_app.py
```

Expected: `5/5 tests passed`.

## Known Runtime Notes

- First startup is slower because TensorFlow/CLIP models load into memory.
- CLIP warnings about `position_ids` shown at load time are expected for this setup.
- Flask debug reloader loads models twice in development mode.

## Start and Stop

Start:

```powershell
python app.py
```

Stop:

- Press `Ctrl + C` in the terminal running Flask.

If you started the server in a background terminal via Copilot tools, stop it by terminating that terminal process.
