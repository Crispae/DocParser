#!/bin/bash

# DocParser Installation Script
# This script sets up the DocParser environment using conda

set -e  # Exit on any error

echo "=== DocParser Installation Script ==="
echo

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Check if environment already exists
if conda env list | grep -q "docparser"; then
    echo "⚠️  DocParser environment already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n docparser -y
    else
        echo "Using existing environment."
        conda activate docparser
        echo "✅ Environment activated."
        echo "Run: python verify_installation.py"
        exit 0
    fi
fi

# Create environment from yml file
echo "Creating DocParser environment..."
if [ -f "environment.yml" ]; then
    conda env create -f environment.yml
else
    echo "environment.yml not found, creating basic environment..."
    conda create -n docparser python=3.9 -y
    conda activate docparser
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    conda install -c conda-forge pillow numpy scikit-learn tqdm -y
    pip install transformers>=4.30.0 PyMuPDF>=1.23.0 pytesseract>=0.3.10 opencv-python>=4.8.0 scikit-image>=0.21.0
fi

# Activate environment
echo "Activating environment..."
conda activate docparser

# Install the package in development mode
echo "Installing DocParser in development mode..."
pip install -e .

echo
echo "✅ Installation completed!"
echo
echo "Next steps:"
echo "1. Verify installation: python verify_installation.py"
echo "2. Try basic usage: python examples/basic_usage.py"
echo "3. Check documentation in README.md"
echo
echo "To activate the environment in the future:"
echo "   conda activate docparser" 