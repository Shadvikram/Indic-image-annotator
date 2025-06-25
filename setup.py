#!/usr/bin/env python3
"""
Setup script for Indic Image Annotator
This script handles initial setup and model downloads
"""

import os
import sys
import subprocess
from pathlib import Path
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def download_models():
    """Download required AI models"""
    print("🤖 Downloading AI models...")
    
    try:
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Download BLIP model
        print("Downloading BLIP model for image captioning...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Save models locally
        processor.save_pretrained("models/blip-processor")
        model.save_pretrained("models/blip-model")
        
        print("✅ BLIP model downloaded successfully")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not download models: {e}")
        print("Models will be downloaded on first use")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = ["temp", "exports", "models", "data", "logs"]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("✅ Directories created successfully")

def create_sample_data():
    """Create sample configuration and data files"""
    print("📋 Creating sample data...")
    
    # Create sample images directory
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    # Create .gitignore
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt

# Models
models/
*.pt
*.pth
*.onnx

# Data
temp/
exports/
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    
    print("✅ Sample data created successfully")

def check_gpu_availability():
    """Check if GPU is available for faster inference"""
    print("🔍 Checking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
    else:
        print("⚠️  No GPU detected. Using CPU (slower inference)")

def create_run_script():
    """Create a script to run the application"""
    run_script_content = """#!/bin/bash
# Run script for Indic Image Annotator

echo "🚀 Starting Indic Image Annotator..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Run the Streamlit app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo "✅ Application started successfully!"
echo "🌐 Open http://localhost:8501 in your browser"
"""
    
    with open("run.sh", "w") as f:
        f.write(run_script_content)
    
    # Make script executable on Unix systems
    if os.name != 'nt':  # Not Windows
        os.chmod("run.sh", 0o755)
    
    # Create Windows batch file
    run_bat_content = """@echo off
echo 🚀 Starting Indic Image Annotator...

REM Activate virtual environment if it exists
if exist venv\\Scripts\\activate.bat (
    echo 📦 Activating virtual environment...
    call venv\\Scripts\\activate.bat
)

REM Run the Streamlit app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo ✅ Application started successfully!
echo 🌐 Open http://localhost:8501 in your browser
pause
"""
    
    with open("run.bat", "w") as f:
        f.write(run_bat_content)
    
    print("✅ Run scripts created successfully")

def main():
    """Main setup function"""
    print("🎯 Setting up Indic Image Annotator...")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Check GPU
    check_gpu_availability()
    
    # Download models (optional)
    try:
        download_models()
    except KeyboardInterrupt:
        print("\n⚠️  Model download interrupted. Models will be downloaded on first use.")
    
    # Create sample data
    create_sample_data()
    
    # Create run scripts
    create_run_script()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the application:")
    print("   • On Linux/Mac: ./run.sh")
    print("   • On Windows: run.bat")
    print("   • Or directly: streamlit run app.py")
    print("\n2. Open http://localhost:8501 in your browser")
    print("\n3. Upload an image and start annotating!")
    print("\n📚 For help, check the README.md file")

if __name__ == "__main__":
    main()