"""
========================================
SETUP.PY - Automated Setup Script
========================================

Purpose: Automate the installation and setup process
Features:
  - Check system requirements
  - Install Python dependencies
  - Verify installation
  - Provide next steps

Usage: python setup.py
========================================
"""

import os
import sys
import subprocess
import platform
import json

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🏥 AI MEDICAL IMAGE CLASSIFICATION - SETUP")
    print("=" * 60)
    print("Automated installation script for BCA project")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    print("\n🔍 Checking Python version...")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        print("Please upgrade Python from https://python.org")
        return False

    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_pip():
    """Check if pip is available"""
    print("\n🔍 Checking pip...")

    try:
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ pip is available")
            return True
        else:
            print("❌ pip not found")
            return False
    except Exception as e:
        print(f"❌ Error checking pip: {e}")
        return False

def install_requirements():
    """Install Python requirements"""
    print("\n📦 Installing Python packages...")
    print("This may take a few minutes...")

    requirements_file = 'requirements.txt'
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found")
        return False

    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Install requirements
        print("Installing requirements...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])

        print("✓ All packages installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed"""
    print("\n🔍 Verifying installation...")

    packages_to_check = [
        'flask',
        'tensorflow',
        'keras',
        'numpy',
        'opencv-python',
        'pillow'
    ]

    failed_packages = []

    for package in packages_to_check:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            else:
                __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_packages.append(package)

    if failed_packages:
        print(f"\n❌ Failed to import: {', '.join(failed_packages)}")
        print("Try reinstalling with: pip install -r requirements.txt")
        return False

    print("✓ All packages verified successfully")
    return True

def check_dataset():
    """Check if dataset is available"""
    print("\n🔍 Checking dataset...")

    dataset_path = os.path.join('data', 'chest_xray', 'train')
    if os.path.exists(dataset_path):
        # Count images
        normal_count = len(os.listdir(os.path.join(dataset_path, 'NORMAL')))
        pneumonia_count = len(os.listdir(os.path.join(dataset_path, 'PNEUMONIA')))

        print(f"✓ Dataset found: {normal_count} NORMAL, {pneumonia_count} PNEUMONIA images")
        return True
    else:
        print("❌ Dataset not found")
        print("Please download from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")
        print("Extract to: data/chest_xray/")
        return False

def check_model():
    """Check if trained model exists"""
    print("\n🔍 Checking trained model...")

    model_path = os.path.join('models', 'model.h5')
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model found: {size_mb:.2f} MB")
        return True
    else:
        print("❌ Model not found (model.h5)")
        print("Run: python train_model.py")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS")
    print("=" * 60)

    steps = [
        ("Download Dataset", "Get from Kaggle and extract to data/chest_xray/"),
        ("Train Model", "Run: python train_model.py (15-30 minutes)"),
        ("Evaluate Model", "Run: python evaluate_model.py (optional)"),
        ("Start Web App", "Run: python app.py"),
        ("Open Browser", "Go to: http://localhost:5000")
    ]

    for i, (title, description) in enumerate(steps, 1):
        print(f"{i}. {title}")
        print(f"   {description}")
        print()

def print_system_info():
    """Print system information"""
    print("\n" + "=" * 60)
    print("💻 SYSTEM INFORMATION")
    print("=" * 60)

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    # Check available memory (rough estimate)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    except ImportError:
        print("RAM: Install psutil to check memory")

def main():
    """Main setup function"""
    print_header()

    # System checks
    if not check_python_version():
        return False

    if not check_pip():
        return False

    print_system_info()

    # Installation
    if not install_requirements():
        return False

    if not verify_installation():
        return False

    # Project checks
    check_dataset()
    check_model()

    # Success message
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Your AI Medical Image Classification system is ready!")
    print("=" * 60)

    print_next_steps()

    return True

if __name__ == '__main__':
    try:
        success = main()
        if success:
            print("\n🎉 Happy learning! Remember: Educational use only!")
        else:
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
