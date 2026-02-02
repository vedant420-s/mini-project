"""
========================================
EVALUATE_MODEL.PY - Model Evaluation Script
========================================

Purpose: Evaluate trained model performance with detailed metrics
Generates confusion matrix and classification reports

Instructions:
1. Ensure model.h5 exists in models/ folder
2. Run: python evaluate_model.py
========================================
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("MODEL EVALUATION SCRIPT")
print("=" * 60)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray', 'test')

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    print("Please run train_model.py first")
    exit(1)

print(f"\n✓ Loading model from {MODEL_PATH}")

# Load the trained model
model = load_model(MODEL_PATH)
print("✓ Model loaded successfully")

# Prepare test data
print("\n" + "=" * 60)
print("PREPARING TEST DATA...")
print("=" * 60)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print(f"✓ Test samples: {test_generator.samples}")

# ============================================
# EVALUATION
# ============================================

print("\n" + "=" * 60)
print("EVALUATING ON TEST SET...")
print("=" * 60)

# Get predictions
test_generator.reset()
predictions = model.predict(test_generator, verbose=0)
y_true = test_generator.classes
y_pred = (predictions > 0.5).astype(int).flatten()

# Calculate metrics
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

print(f"\n✓ Test Loss: {test_loss:.4f}")
print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ============================================
# CONFUSION MATRIX
# ============================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_true, y_pred)
print(f"\n{cm}")
print(f"\nTrue Negatives (NORMAL correctly identified): {cm[0,0]}")
print(f"False Positives (NORMAL incorrectly as PNEUMONIA): {cm[0,1]}")
print(f"False Negatives (PNEUMONIA incorrectly as NORMAL): {cm[1,0]}")
print(f"True Positives (PNEUMONIA correctly identified): {cm[1,1]}")

# ============================================
# CLASSIFICATION REPORT
# ============================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(
    y_true,
    y_pred,
    target_names=['NORMAL', 'PNEUMONIA'],
    digits=4
))

# ============================================
# METRICS EXPLANATION
# ============================================

print("\n" + "=" * 60)
print("METRICS EXPLANATION FOR BCA STUDENTS")
print("=" * 60)

print("""
PRECISION: How many predictions of PNEUMONIA were actually correct?
   - High precision = Few false positives (we don't scare healthy people)

RECALL: Of all actual PNEUMONIA cases, how many did we catch?
   - High recall = Few false negatives (we don't miss sick people)

F1-SCORE: Balance between precision and recall
   - Harmonic mean of precision and recall
   - Use when both metrics are important

SUPPORT: Number of test samples in each class

CONFUSION MATRIX:
   - Shows correct vs incorrect predictions
   - True Positives: Correctly identified PNEUMONIA
   - True Negatives: Correctly identified NORMAL
   - False Positives: Incorrectly marked as PNEUMONIA
   - False Negatives: Incorrectly marked as NORMAL (DANGEROUS!)

In medical diagnosis:
   - High recall is CRITICAL (don't miss sick people)
   - False negatives are dangerous (missing pneumonia)
   - Some false positives are acceptable (second opinion possible)
""")

print("=" * 60)
