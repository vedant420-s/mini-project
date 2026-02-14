"""
========================================
APP.PY - FLASK WEB APPLICATION
========================================

Author: Your Name
Purpose: Web API for medical image classification using trained CNN
Features:
  - Image upload endpoint
  - Image preprocessing
  - Model prediction
  - Confidence percentage
  - Medical disclaimer

Instructions:
1. Ensure model.h5 exists in models/ folder
2. Install requirements: pip install -r requirements.txt
3. Run: python app.py
4. Open browser: http://localhost:5000

========================================
"""

from flask import Flask, render_template, request, jsonify
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import json

# ============================================
# FLASK APP INITIALIZATION
# ============================================

print("=" * 60)
print("INITIALIZING FLASK APP...")
print("=" * 60)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ============================================
# LOAD MODELS
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PNEUMONIA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xray_detector.h5')

print(f"Loading pneumonia model from {PNEUMONIA_MODEL_PATH}...")
print(f"Loading detector model from {DETECTOR_MODEL_PATH}...")

# Check if pneumonia model exists
if not os.path.exists(PNEUMONIA_MODEL_PATH):
    print(f"❌ ERROR: Pneumonia model not found at {PNEUMONIA_MODEL_PATH}")
    print("Please run train_model.py first to train the pneumonia model")
    exit(1)

# Check if detector model exists
if not os.path.exists(DETECTOR_MODEL_PATH):
    print(f"⚠️ WARNING: Detector model not found at {DETECTOR_MODEL_PATH}")
    print("Running without detector model - assuming all images are chest X-rays")
    detector_model = None

try:
    # Load the trained pneumonia model
    pneumonia_model = load_model(PNEUMONIA_MODEL_PATH)
    print("✓ Pneumonia model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading pneumonia model: {e}")
    exit(1)

if detector_model is not None:
    try:
        # Load the trained detector model
        detector_model = load_model(DETECTOR_MODEL_PATH)
        print("✓ Detector model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading detector model: {e}")
        detector_model = None

# ============================================
# DEFINE CLASSES
# ============================================

# Class names for prediction
CLASS_NAMES = {
    0: 'NORMAL',
    1: 'PNEUMONIA'
}

# Class descriptions (educational)
CLASS_DESCRIPTIONS = {
    'NORMAL': 'The X-ray appears normal with no signs of pneumonia detected.',
    'PNEUMONIA': 'The X-ray shows signs consistent with pneumonia. Medical consultation recommended.'
}

print("\n✓ Model configuration:")
print(f"  - Input size: 224x224 pixels")
print(f"  - Classes: {CLASS_NAMES}")
print(f"  - Model type: MobileNetV2 + Custom Dense Layers")

# ============================================
# HELPER FUNCTIONS
# ============================================

def preprocess_image(img_file):
    """
    Preprocess image for model prediction
    
    Args:
        img_file: Flask file object from request
    
    Returns:
        Preprocessed numpy array ready for prediction
    """
    try:
        # Read image file
        img = Image.open(img_file.stream)
        
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size (224x224)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize pixel values to 0-1
        img_array = img_array / 255.0
        
        # Add batch dimension (model expects batch of images)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, True, "Image preprocessed successfully"
    
    except Exception as e:
        return None, False, f"Error preprocessing image: {str(e)}"

def detect_chest_xray(img_array):
    """
    Detect if the image is a chest X-ray using the detector model

    Args:
        img_array: Preprocessed numpy array

    Returns:
        is_chest_xray (bool), confidence (float), success (bool), message (str)
    """
    # If detector model is not available, assume all images are chest X-rays
    if detector_model is None:
        return True, 1.0, True, "Detector model not available - assuming valid chest X-ray"

    try:
        # Get detector model prediction (output is probability for class 1 = Chest X-ray)
        prediction = detector_model.predict(img_array, verbose=0)

        # Extract confidence (probability of being a chest X-ray)
        confidence = prediction[0][0]

        # Determine if it's a chest X-ray (>90% confidence threshold)
        is_chest_xray = confidence > 0.9

        return is_chest_xray, confidence, True, "Detection successful"

    except Exception as e:
        return False, 0.0, False, f"Error in detection: {str(e)}"

def predict_pneumonia(img_array):
    """
    Make pneumonia prediction on preprocessed chest X-ray image

    Args:
        img_array: Preprocessed numpy array

    Returns:
        Prediction class and confidence
    """
    try:
        # Get pneumonia model prediction (output is probability for class 1 = PNEUMONIA)
        prediction = pneumonia_model.predict(img_array, verbose=0)

        # Extract confidence (probability)
        confidence = prediction[0][0]

        # Determine class (0 if confidence < 0.5, else 1)
        class_idx = 1 if confidence > 0.5 else 0

        # Adjust confidence to represent the predicted class
        if class_idx == 0:
            confidence = 1 - confidence

        return class_idx, confidence, True, "Pneumonia prediction successful"

    except Exception as e:
        return None, None, False, f"Error making pneumonia prediction: {str(e)}"

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and return prediction
    
    Expected request:
        - POST multipart/form-data with 'file' field containing image
    
    Returns:
        JSON with prediction results or error message
    """
    print("\n" + "=" * 60)
    print("PREDICTION REQUEST RECEIVED")
    print("=" * 60)
    
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            print("❌ No file provided in request")
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        img_file = request.files['file']
        
        # Check if file is selected
        if img_file.filename == '':
            print("❌ No file selected")
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file extension
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
        if not ('.' in img_file.filename and 
                img_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            print("❌ Invalid file format")
            return jsonify({
                'success': False,
                'error': 'Invalid file format. Please upload JPG, PNG, GIF, or BMP'
            }), 400
        
        print(f"✓ File received: {img_file.filename}")
        
        # ============================================
        # STEP 1: PREPROCESS IMAGE
        # ============================================
        
        img_file.stream.seek(0)  # Reset file pointer
        img_array, preprocess_success, preprocess_msg = preprocess_image(img_file)
        
        if not preprocess_success:
            print(f"❌ Preprocessing failed: {preprocess_msg}")
            return jsonify({
                'success': False,
                'error': preprocess_msg
            }), 400
        
        print(f"✓ {preprocess_msg}")
        
        # ============================================
        # STEP 2: DETECT CHEST X-RAY (GATEKEEPER)
        # ============================================

        is_chest_xray, detector_confidence, detect_success, detect_msg = detect_chest_xray(img_array)

        if not detect_success:
            print(f"❌ Detection failed: {detect_msg}")
            return jsonify({
                'success': False,
                'error': detect_msg
            }), 500

        print(f"✓ {detect_msg}")
        detector_confidence_percent = round(float(detector_confidence) * 100, 2)
        print(f"✓ Detector confidence: {detector_confidence_percent}%")

        # Check if image is a chest X-ray with >90% confidence
        if not is_chest_xray:
            print(f"❌ Not a chest X-ray (confidence: {detector_confidence_percent}%)")
            return jsonify({
                'success': False,
                'error': 'Uploaded image is not a valid Chest X-ray. Please upload a Chest X-ray image.'
            }), 400

        print("✓ Image confirmed as chest X-ray - proceeding to pneumonia analysis")

        # ============================================
        # STEP 3: PNEUMONIA PREDICTION
        # ============================================

        class_idx, confidence, pred_success, pred_msg = predict_pneumonia(img_array)

        if not pred_success:
            print(f"❌ Pneumonia prediction failed: {pred_msg}")
            return jsonify({
                'success': False,
                'error': pred_msg
            }), 500

        print(f"✓ {pred_msg}")

        # ============================================
        # STEP 4: FORMAT RESPONSE
        # ============================================

        class_name = CLASS_NAMES[class_idx]
        confidence_percent = round(float(confidence) * 100, 2)

        print(f"✓ Prediction: {class_name}")
        print(f"✓ Confidence: {confidence_percent}%")

        # Prepare response
        response = {
            'success': True,
            'prediction': class_name,
            'confidence': confidence_percent,
            'description': CLASS_DESCRIPTIONS[class_name],
            'message': f"{confidence_percent}% confidence that the X-ray shows {class_name}",
            'detector_confidence': detector_confidence_percent
        }

        print("✓ Response prepared successfully")
        print("=" * 60)

        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Unexpected error: {str(e)}"
        }), 500

@app.route('/about')
def about():
    """
    Return information about the application
    """
    about_info = {
        'name': 'AI Medical Image Classification System',
        'description': 'CNN-based classification of chest X-rays for pneumonia detection',
        'model': 'MobileNetV2 with Transfer Learning',
        'dataset': 'Chest X-ray Pneumonia (Kaggle)',
        'classes': CLASS_NAMES,
        'disclaimer': 'This system is for educational purposes only. It should NOT be used for actual medical diagnosis.'
    }
    return jsonify(about_info), 200

@app.route('/documentation')
def documentation():
    """
    Display project documentation page
    """
    return render_template('documentation.html')

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File is too large. Maximum size is 16MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("STARTING FLASK SERVER")
    print("=" * 60)
    print("\n✓ Server starting...")
    print("✓ Open your browser and go to: http://localhost:5000")
    print("✓ Press Ctrl+C to stop the server")
    print("\n" + "=" * 60 + "\n")
    
    # Run Flask development server
    # debug=True: Auto-reload on code changes, detailed error messages
    app.run(debug=True, host='localhost', port=5000)
