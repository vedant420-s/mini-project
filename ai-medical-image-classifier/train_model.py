"""
========================================
TRAIN_MODEL.PY - CNN Model Training Script
========================================

Author: Your Name
Purpose: Train a CNN model using Transfer Learning (MobileNetV2)
Dataset: Kaggle Chest X-ray Pneumonia Dataset (NORMAL vs PNEUMONIA)
Output: Saves trained model as 'models/model.h5'

Key Concepts:
- Transfer Learning: Uses pre-trained MobileNetV2 weights
- Data Augmentation: Enhances training data with random transformations
- CPU Friendly: Uses MobileNetV2 (lightweight model)
- Batch Processing: Efficient memory usage

Instructions:
1. Download dataset from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
2. Extract to: data/chest_xray/
3. Run: python train_model.py
4. Wait for training to complete (~15-30 minutes on CPU)
========================================
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ============================================
# STEP 1: CONFIGURATION
# ============================================

print("=" * 60)
print("INITIALIZING CONFIGURATION...")
print("=" * 60)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')

# Model parameters
IMG_SIZE = 224  # MobileNetV2 requires 224x224 images
BATCH_SIZE = 32
EPOCHS = 3  # Reduced for faster training
LEARNING_RATE = 0.001

print(f"✓ Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"✓ Batch Size: {BATCH_SIZE}")
print(f"✓ Epochs: {EPOCHS}")
print(f"✓ Learning Rate: {LEARNING_RATE}")

# ============================================
# STEP 2: CHECK DATA DIRECTORY
# ============================================

print("\n" + "=" * 60)
print("CHECKING DATA DIRECTORY...")
print("=" * 60)

if not os.path.exists(TRAIN_DIR):
    print(f"❌ ERROR: Training data not found at {TRAIN_DIR}")
    print("\nPlease download the dataset from:")
    print("https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")
    print("\nAnd extract it to the 'data/chest_xray/' folder.")
    exit(1)

print(f"✓ Data directory found: {TRAIN_DIR}")

# Count images in each class
train_normal = len(os.listdir(os.path.join(TRAIN_DIR, 'NORMAL')))
train_pneumonia = len(os.listdir(os.path.join(TRAIN_DIR, 'PNEUMONIA')))
test_normal = len(os.listdir(os.path.join(TEST_DIR, 'NORMAL')))
test_pneumonia = len(os.listdir(os.path.join(TEST_DIR, 'PNEUMONIA')))

print(f"\nDataset Statistics:")
print(f"  Train - NORMAL: {train_normal}, PNEUMONIA: {train_pneumonia}")
print(f"  Test  - NORMAL: {test_normal}, PNEUMONIA: {test_pneumonia}")

# ============================================
# STEP 3: DATA PREPROCESSING & AUGMENTATION
# ============================================

print("\n" + "=" * 60)
print("PREPARING DATA WITH AUGMENTATION...")
print("=" * 60)

# Data Augmentation for training (prevents overfitting)
# These transformations make the model more robust
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to 0-1
    rotation_range=20,           # Random rotation up to 20 degrees
    width_shift_range=0.2,       # Random horizontal shift
    height_shift_range=0.2,      # Random vertical shift
    shear_range=0.2,             # Random shearing
    zoom_range=0.2,              # Random zoom
    horizontal_flip=True,        # Random horizontal flip
    fill_mode='nearest'          # Fill empty pixels
)

# No augmentation for testing (just normalize)
test_datagen = ImageDataGenerator(
    rescale=1./255               # Only normalize
)

# Load training data in batches
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Binary classification: 0=NORMAL, 1=PNEUMONIA
)

# Load testing data in batches
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print(f"✓ Training samples: {train_generator.samples}")
print(f"✓ Testing samples: {test_generator.samples}")
print(f"✓ Data augmentation applied successfully")

# ============================================
# STEP 4: BUILD MODEL WITH TRANSFER LEARNING
# ============================================

print("\n" + "=" * 60)
print("BUILDING MODEL WITH TRANSFER LEARNING...")
print("=" * 60)

# Load pre-trained MobileNetV2 (trained on ImageNet)
print("Loading pre-trained MobileNetV2 weights...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,  # Remove classification layer
    weights='imagenet'  # Use ImageNet pre-trained weights
)

# Freeze base model weights (don't train them)
# This speeds up training and prevents overfitting
base_model.trainable = False

print(f"✓ MobileNetV2 loaded with {len(base_model.layers)} layers")
print(f"✓ Base model weights frozen")

# Build custom model on top of MobileNetV2
model = models.Sequential([
    # Input layer
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Pre-trained MobileNetV2 base
    base_model,
    
    # Custom classification layers
    layers.GlobalAveragePooling2D(),     # Pool features to 1D vector
    layers.Dense(256, activation='relu'), # Fully connected layer
    layers.Dropout(0.5),                  # Dropout (prevents overfitting)
    layers.Dense(128, activation='relu'), # Another fully connected layer
    layers.Dropout(0.5),                  # Dropout
    layers.Dense(1, activation='sigmoid') # Output layer (binary classification)
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',     # Loss function for binary classification
    metrics=['accuracy']            # Metric to track
)

print("\n✓ Model built successfully!")
print("\nModel Architecture:")
model.summary()

# ============================================
# STEP 5: TRAIN THE MODEL
# ============================================

print("\n" + "=" * 60)
print("TRAINING MODEL...")
print("=" * 60)
print("This may take 15-30 minutes on CPU...")
print("=" * 60)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=test_generator,
    verbose=1
)

print("\n✓ Training completed!")

# ============================================
# STEP 6: EVALUATE MODEL
# ============================================

print("\n" + "=" * 60)
print("EVALUATING MODEL ON TEST DATA...")
print("=" * 60)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print(f"\n✓ Test Loss: {test_loss:.4f}")
print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Get predictions for detailed metrics
test_generator.reset()
predictions = model.predict(test_generator, verbose=0)
y_true = test_generator.classes
y_pred = (predictions > 0.5).astype(int).flatten()

# Classification report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(
    y_true, 
    y_pred,
    target_names=['NORMAL', 'PNEUMONIA']
))

# ============================================
# STEP 7: SAVE MODEL
# ============================================

print("\n" + "=" * 60)
print("SAVING MODEL...")
print("=" * 60)

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model in HDF5 format
model.save(MODEL_PATH)

print(f"✓ Model saved at: {MODEL_PATH}")
print(f"✓ File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")

# ============================================
# STEP 8: SAVE TRAINING HISTORY
# ============================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

# Display final metrics
print(f"\nFinal Metrics:")
print(f"  Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Train Loss: {history.history['loss'][-1]:.4f}")
print(f"  Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"  Val Loss: {history.history['val_loss'][-1]:.4f}")

print(f"\n✓ Model is ready for inference!")
print(f"✓ The Flask app (app.py) will load this model automatically.")
print(f"✓ Run: python app.py")

# ============================================
# NOTES FOR BCA STUDENTS
# ============================================
print("\n" + "=" * 60)
print("KEY CONCEPTS EXPLAINED")
print("=" * 60)
print("""
1. TRANSFER LEARNING:
   - Uses pre-trained MobileNetV2 (trained on millions of images)
   - Much faster and accurate than training from scratch
   - Requires less data and computational power

2. DATA AUGMENTATION:
   - Creates variations of training images (rotate, flip, zoom, etc.)
   - Prevents the model from memorizing specific images
   - Improves generalization to new images

3. DROPOUT:
   - Randomly disables some neurons during training
   - Prevents overfitting (memorization)
   - Makes the model more robust

4. BATCH PROCESSING:
   - Processes multiple images at once
   - Efficient memory usage
   - Faster training

5. BINARY CLASSIFICATION:
   - Output is 0 (NORMAL) or 1 (PNEUMONIA)
   - Uses Sigmoid activation (probability between 0-1)
   - Uses Binary Crossentropy loss
""")

print("=" * 60)
