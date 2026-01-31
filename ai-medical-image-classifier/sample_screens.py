"""
========================================
SAMPLE_SCREENS.PY - Generate Sample Screenshots
========================================

Purpose: Create placeholder images for documentation
Features:
  - Generate sample X-ray style images
  - Create UI mockups
  - Save as PNG files

Usage: python sample_screens.py
========================================
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_directories():
    """Create screenshots directory"""
    os.makedirs('screenshots', exist_ok=True)
    print("✓ Created screenshots directory")

def create_sample_xray():
    """Create a sample X-ray style image"""
    print("Creating sample X-ray image...")

    # Create a grayscale image with some structure
    img = Image.new('L', (224, 224), color=128)

    # Add some random noise to simulate X-ray texture
    pixels = np.array(img)
    noise = np.random.normal(0, 25, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)

    # Add some circular shapes to simulate bones/lungs
    img_array = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img_array)

    # Draw some anatomical structures
    draw.ellipse([50, 50, 150, 150], fill=200, outline=255)
    draw.ellipse([80, 80, 120, 120], fill=150, outline=200)

    # Save as sample X-ray
    img_array.save('screenshots/sample_xray.png')
    print("✓ Saved sample_xray.png")

def create_ui_mockup():
    """Create a simple UI mockup"""
    print("Creating UI mockup...")

    # Create a white background
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)

    try:
        # Try to use a default font
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Header
    draw.rectangle([0, 0, 800, 80], fill='#667eea')
    draw.text((20, 25), "🏥 Medical Image Classification System", fill='white', font=font)

    # Disclaimer box
    draw.rectangle([20, 100, 760, 180], fill='#fff3cd', outline='#ff6b6b')
    draw.text((30, 110), "⚠️ MEDICAL DISCLAIMER", fill='#c92a2a', font=font)
    draw.text((30, 140), "This is for educational purposes only. Not for medical diagnosis.", fill='#333', font=small_font)

    # Upload area
    draw.rectangle([20, 200, 760, 350], fill='#f8f9fa', outline='#667eea')
    draw.text((300, 250), "📁 Drag & Drop X-ray Image Here", fill='#667eea', font=font)
    draw.text((250, 280), "or click to browse (JPG, PNG, GIF, BMP)", fill='#999', font=small_font)

    # Results area (placeholder)
    draw.rectangle([20, 380, 760, 520], fill='#d4edda', outline='#28a745')
    draw.text((30, 390), "✓ PREDICTION RESULT", fill='#155724', font=font)
    draw.text((30, 430), "Classification: NORMAL", fill='#333', font=small_font)
    draw.text((30, 450), "Confidence: 94.2%", fill='#333', font=small_font)
    draw.text((30, 480), "⚠️ Always consult a medical professional", fill='#c92a2a', font=small_font)

    # Save mockup
    img.save('screenshots/ui_mockup.png')
    print("✓ Saved ui_mockup.png")

def create_training_progress():
    """Create a training progress visualization"""
    print("Creating training progress chart...")

    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Title
    draw.text((200, 20), "Model Training Progress", fill='#333', font=font)

    # Mock training curves
    # Accuracy
    draw.line([(50, 350), (100, 320), (150, 290), (200, 270), (250, 250), (300, 240), (350, 235), (400, 230), (450, 228), (500, 225)], fill='#28a745', width=3)
    draw.text((520, 220), "Accuracy", fill='#28a745', font=small_font)

    # Loss
    draw.line([(50, 100), (100, 120), (150, 140), (200, 150), (250, 155), (300, 158), (350, 160), (400, 162), (450, 163), (500, 164)], fill='#dc3545', width=3)
    draw.text((520, 160), "Loss", fill='#dc3545', font=small_font)

    # Axes labels
    draw.text((280, 370), "Epoch", fill='#666', font=small_font)
    draw.text((10, 200), "Value", fill='#666', font=small_font)

    # Grid lines
    for i in range(1, 10):
        x = 50 + i * 50
        draw.line([(x, 80), (x, 350)], fill='#eee', width=1)
        draw.line([(50, 80 + i * 30), (550, 80 + i * 30)], fill='#eee', width=1)

    img.save('screenshots/training_progress.png')
    print("✓ Saved training_progress.png")

def create_readme_images():
    """Create images referenced in README"""
    print("Creating README placeholder images...")

    # Architecture diagram placeholder
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    draw.text((200, 180), "CNN Architecture Diagram", fill='#667eea', font=font)
    draw.text((150, 220), "(Placeholder - Create actual diagram)", fill='#999', font=font)
    img.save('screenshots/architecture.png')
    print("✓ Saved architecture.png")

def main():
    """Main function"""
    print("=" * 60)
    print("🎨 SAMPLE SCREENS GENERATOR")
    print("=" * 60)

    create_directories()
    create_sample_xray()
    create_ui_mockup()
    create_training_progress()
    create_readme_images()

    print("\n" + "=" * 60)
    print("✅ SAMPLE SCREENS CREATED!")
    print("=" * 60)
    print("Generated files in screenshots/ folder:")
    print("  - sample_xray.png")
    print("  - ui_mockup.png")
    print("  - training_progress.png")
    print("  - architecture.png")
    print("\nUse these for documentation and presentations!")

if __name__ == '__main__':
    main()
