"""
========================================
DOWNLOAD_DIVERSE_XRAYS.PY
========================================

Purpose: Download diverse X-ray images (foot, hand, spine, etc.) from public sources
to improve detector model training by adding more non-chest X-ray examples

Sources:
- MURA (Musculoskeletal Radiographs) - foot, hand, shoulder, elbow, etc.
- Kaggle Datasets - various X-ray types
- Open medical imaging repositories

Instructions:
1. Run: python download_diverse_xrays.py
2. Wait for downloads to complete
3. Re-run: python train_detector.py (with class weights)
========================================
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import shutil
import time

# Create directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'detector_dataset')
NOT_CHEST_DIR_TRAIN = os.path.join(DATA_DIR, 'train', 'not_chest_xray')
NOT_CHEST_DIR_TEST = os.path.join(DATA_DIR, 'test', 'not_chest_xray')

os.makedirs(NOT_CHEST_DIR_TRAIN, exist_ok=True)
os.makedirs(NOT_CHEST_DIR_TEST, exist_ok=True)

print("=" * 60)
print("DOWNLOADING DIVERSE X-RAY IMAGES")
print("=" * 60)

# URLs to diverse X-ray images from public sources
# These are sample X-rays from different body parts

diverse_xray_urls = {
    # Foot X-rays
    'foot_1': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Foot_X-ray.jpg/640px-Foot_X-ray.jpg',
    'foot_2': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Ankle_x-ray.jpg/640px-Ankle_x-ray.jpg',
    
    # Hand/Wrist X-rays
    'hand_1': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Hand_X-ray.jpg/640px-Hand_X-ray.jpg',
    'hand_2': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Wrist_X-ray.jpg/640px-Wrist_X-ray.jpg',
    
    # Spine X-rays
    'spine_1': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Lumbar_X-ray.jpg/640px-Lumbar_X-ray.jpg',
    'spine_2': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Cervical_spine_X-ray.jpg/640px-Cervical_spine_X-ray.jpg',
    
    # Pelvis X-rays
    'pelvis_1': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Pelvis_X-ray.jpg/640px-Pelvis_X-ray.jpg',
    'pelvis_2': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Hip_X-ray.jpg/640px-Hip_X-ray.jpg',
    
    # Elbow/Arm X-rays
    'arm_1': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Elbow_X-ray.jpg/640px-Elbow_X-ray.jpg',
    'arm_2': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Arm_X-ray.jpg/640px-Arm_X-ray.jpg',
    
    # Knee X-rays
    'knee_1': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Knee_X-ray.jpg/640px-Knee_X-ray.jpg',
    'knee_2': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Knee_lateral_X-ray.jpg/640px-Knee_lateral_X-ray.jpg',
}

def create_session():
    """Create a requests session with retries"""
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def download_image(url, filepath, timeout=10):
    """Download an image from URL"""
    try:
        session = create_session()
        response = session.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ❌ Failed to download from {url}: {str(e)}")
        return False

# Download images
downloaded_count = 0
failed_count = 0

print("\nDownloading diverse X-ray images...")
print("-" * 60)

for img_name, url in diverse_xray_urls.items():
    print(f"\nDownloading {img_name}...")
    
    # Save to training set
    train_filepath = os.path.join(NOT_CHEST_DIR_TRAIN, f"{img_name}.jpg")
    
    if download_image(url, train_filepath):
        print(f"  ✓ Downloaded to train set")
        downloaded_count += 1
        
        # Optionally copy to test set
        if downloaded_count % 3 == 0:  # Put ~1/3 in test set
            test_filepath = os.path.join(NOT_CHEST_DIR_TEST, f"{img_name}.jpg")
            try:
                shutil.copy(train_filepath, test_filepath)
                print(f"  ✓ Also added to test set")
            except:
                pass
    else:
        failed_count += 1

print("\n" + "=" * 60)
print(f"Download Summary:")
print(f"  ✓ Successfully downloaded: {downloaded_count} images")
print(f"  ❌ Failed: {failed_count} images")
print(f"  📁 Train set: {len(os.listdir(NOT_CHEST_DIR_TRAIN))} images")
print(f"  📁 Test set: {len(os.listdir(NOT_CHEST_DIR_TEST))} images")
print("=" * 60)

print("\n" + "=" * 60)
print("GENERATING SYNTHETIC IMAGES (if PIL available)")
print("=" * 60)

try:
    from PIL import Image, ImageDraw, ImageFilter
    import random
    
    print("\nGenerating synthetic X-ray-like images...")
    
    # Generate synthetic non-chest X-ray patterns
    for i in range(30):  # Generate 30 synthetic images
        # Create a random pattern that looks X-ray-like
        img = Image.new('L', (224, 224), color=0)
        draw = ImageDraw.Draw(img)
        
        # Draw random shapes to mimic X-ray patterns
        for _ in range(random.randint(5, 15)):
            x1 = random.randint(0, 200)
            y1 = random.randint(0, 200)
            x2 = x1 + random.randint(10, 100)
            y2 = y1 + random.randint(10, 100)
            intensity = random.randint(100, 255)
            draw.rectangle([x1, y1, x2, y2], fill=intensity)
        
        # Add some blur to make it look more X-ray-like
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Save to training set
        train_path = os.path.join(NOT_CHEST_DIR_TRAIN, f"synthetic_{i:03d}.jpg")
        img.save(train_path)
        
        # Save some to test set
        if i % 3 == 0:
            test_path = os.path.join(NOT_CHEST_DIR_TEST, f"synthetic_{i:03d}.jpg")
            img.save(test_path)
    
    print(f"  ✓ Generated 30 synthetic X-ray patterns")
    
except ImportError:
    print("  ⚠ PIL not available, skipping synthetic image generation")

print("\n" + "=" * 60)
print(f"FINAL STATISTICS")
print("=" * 60)
train_count = len(os.listdir(NOT_CHEST_DIR_TRAIN))
test_count = len(os.listdir(NOT_CHEST_DIR_TEST))

print(f"  📁 Non-chest training samples: {train_count}")
print(f"  📁 Non-chest test samples: {test_count}")
print("\nNext steps:")
print("  1. Run: python train_detector.py")
print("     (This will retrain with improved dataset)")
print("=" * 60)
