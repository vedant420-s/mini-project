"""Rule-based AI logic used by CuraVision CDSS pages."""


def generate_prediction_symptoms(prediction):
    """Generate simple symptoms summary from image prediction."""
    if prediction == "PNEUMONIA":
        return "cough, fever, chest pain, breathing difficulty"
    return "No major symptoms detected"


def build_ai_insights_summary(prediction, confidence, manual_symptoms, generated_symptoms):
    """Build readable AI insights for case detail view."""
    return (
        f"Prediction suggests {prediction} with confidence {confidence:.2f}%. "
        f"Reported symptoms: {manual_symptoms}. "
        f"AI-generated symptom pattern: {generated_symptoms}."
    )


def ai_helper_assessment(patient_name, age, symptoms, prediction_result):
    """Rule-based clinical decision support output.

    This is educational logic and not a medical diagnosis.
    """
    text = (symptoms or "").lower()
    prediction_upper = (prediction_result or "").upper()

    respiratory_flags = ["fever", "cough", "breath", "chest pain", "shortness", "fatigue"]
    respiratory_count = sum(1 for flag in respiratory_flags if flag in text)

    if prediction_upper == "PNEUMONIA":
        if respiratory_count >= 3:
            risk = "High"
            recommendation = "Immediate medical attention"
        else:
            risk = "Medium"
            recommendation = "Consult doctor"
        condition = "Possible Pneumonia"
    elif prediction_upper == "NORMAL":
        if respiratory_count >= 3:
            risk = "Medium"
            recommendation = "Consult doctor"
            condition = "Respiratory symptoms despite normal scan"
        else:
            risk = "Low"
            recommendation = "Monitor"
            condition = "No strong signs of severe respiratory disease"
    else:
        if respiratory_count >= 3:
            risk = "Medium"
            recommendation = "Consult doctor"
            condition = "Possible respiratory condition"
        else:
            risk = "Low"
            recommendation = "Monitor"
            condition = "Unclear condition from provided data"

    summary = (
        f"For patient {patient_name}, age {age}, the AI helper flags {condition.lower()} "
        f"with {risk.lower()} risk based on symptoms and prediction context."
    )

    return {
        "possible_condition": condition,
        "risk_level": risk,
        "recommendation": recommendation,
        "summary": summary,
    }
