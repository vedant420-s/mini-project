"""
========================================
DOWNLOAD_DATASET.PY - Dataset Downloader Script
========================================

Purpose: Download and extract the Kaggle Chest X-ray dataset
Features:
  - Automated download from Kaggle
  - Automatic extraction
  - Folder structure creation
  - Progress tracking

Requirements:
  - kaggle API key (kaggle.json)
  - Internet connection

Usage: python download_dataset.py
========================================
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
import shutil

def print_header():
    """Print script header"""
    print("=" * 60)
    print("📥 DATASET DOWNLOADER")
    print("=" * 60)
    print("Automated download of Chest X-ray Pneumonia dataset")
    print("=" * 60)

def check_kaggle_setup():
    """Check if Kaggle API is set up"""
    print("\n🔍 Checking Kaggle API setup...")

    # Check if kaggle is installed
    try:
        import kaggle
        print("✓ Kaggle package installed")
    except ImportError:
        print("❌ Kaggle package not installed")
        print("Install with: pip install kaggle")
        return False

    # Check for kaggle.json
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'

    if kaggle_json.exists():
        print("✓ Kaggle API key found")
        return True
    else:
        print("❌ Kaggle API key not found")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to:", kaggle_dir)
        print("4. Or run: kaggle competitions download -c chest-xray-pneumonia")
        return False

def download_dataset():
    """Download the dataset using Kaggle API"""
    print("\n📥 Downloading dataset...")
    print("This may take several minutes depending on your internet speed...")

    try:
        import kaggle

        # Set up data directory
        data_dir = Path('data/chest_xray')
        data_dir.mkdir(parents=True, exist_ok=True)

        # Change to data directory
        os.chdir(data_dir)

        # Download command
        os.system('kaggle datasets download -d paultimothymooney/chest-xray-pneumonia')

        print("✓ Download completed")
        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def extract_dataset():
    """Extract the downloaded ZIP file"""
    print("\n📦 Extracting dataset...")

    zip_file = Path('chest-xray-pneumonia.zip')

    if not zip_file.exists():
        print("❌ ZIP file not found")
        return False

    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('.')

        print("✓ Extraction completed")

        # Remove ZIP file
        zip_file.unlink()
        print("✓ ZIP file cleaned up")

        return True

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def verify_dataset():
    """Verify the extracted dataset structure"""
    print("\n🔍 Verifying dataset structure...")

    required_dirs = [
        'chest_xray/train/NORMAL',
        'chest_xray/train/PNEUMONIA',
        'chest_xray/test/NORMAL',
        'chest_xray/test/PNEUMONIA'
    ]

    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            # Count files
            file_count = len([f for f in os.listdir(dir_path) if f.endswith(('.jpeg', '.jpg', '.png'))])
            print(f"✓ {dir_path}: {file_count} images")
        else:
            print(f"❌ Missing directory: {dir_path}")
            return False

    return True

def manual_download_instructions():
    """Show manual download instructions"""
    print("\n" + "=" * 60)
    print("📋 MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("If automated download fails, download manually:")
    print()
    print("1. Go to: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")
    print("2. Click 'Download' button")
    print("3. Download 'chest-xray-pneumonia.zip'")
    print("4. Extract to: data/chest_xray/")
    print("5. The folder should contain:")
    print("   - chest_xray/")
    print("     ├── train/")
    print("     │   ├── NORMAL/     (1,341 images)")
    print("     │   └── PNEUMONIA/  (3,875 images)")
    print("     └── test/")
    print("         ├── NORMAL/     (234 images)")
    print("         └── PNEUMONIA/  (390 images)")
    print()
    print("Expected total: ~5,840 X-ray images")

def main():
    """Main function"""
    print_header()

    # Check if dataset already exists
    if os.path.exists('data/chest_xray/train/NORMAL'):
        print("✓ Dataset already exists!")
        if verify_dataset():
            print("✓ Dataset is complete and ready to use")
            return True
        else:
            print("❌ Dataset appears incomplete")

    # Try automated download
    if check_kaggle_setup():
        if download_dataset():
            # Go back to project root
            os.chdir('../..')

            if extract_dataset():
                if verify_dataset():
                    print("\n✅ DATASET DOWNLOAD COMPLETED!")
                    print("You can now run: python train_model.py")
                    return True

    # Fallback to manual instructions
    print("\n❌ Automated download failed")
    manual_download_instructions()

    return False

if __name__ == '__main__':
    try:
        success = main()
        if success:
            print("\n🎉 Dataset ready! Happy training!")
        else:
            print("\n❌ Dataset download failed. Please try manual method.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
