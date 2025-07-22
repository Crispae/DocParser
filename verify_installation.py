#!/usr/bin/env python3
"""
DocParser Installation Verification Script

This script verifies that all required dependencies are properly installed
and the DocParser library is ready to use.
"""

import sys
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Check if Python version meets requirements"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    
    print("✅ Python version is compatible")
    return True


def check_pytorch():
    """Check PyTorch installation and CUDA availability"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"✅ CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - will use CPU")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False


def check_transformers():
    """Check transformers library"""
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        return True
    except ImportError:
        print("❌ Transformers not installed")
        return False


def check_pillow():
    """Check PIL/Pillow installation"""
    try:
        from PIL import Image
        print(f"✅ Pillow installed")
        return True
    except ImportError:
        print("❌ Pillow not installed")
        return False


def check_pymupdf():
    """Check PyMuPDF installation"""
    try:
        import fitz
        print(f"✅ PyMuPDF installed")
        return True
    except ImportError:
        print("❌ PyMuPDF not installed")
        return False


def check_docparser():
    """Check DocParser library"""
    try:
        from DocParser import DocParser, OCREngine
        print("✅ DocParser imported successfully")
        
        # Check available engines
        engines = DocParser.get_available_engines()
        print(f"✅ Available engines: {len(engines)}")
        for engine in engines:
            info = DocParser.get_engine_info(engine)
            print(f"   - {info['name']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ DocParser import error: {e}")
        return False
    except Exception as e:
        print(f"❌ DocParser error: {e}")
        return False


def check_optional_dependencies():
    """Check optional dependencies"""
    optional_deps = {
        'opencv-python': 'cv2',
        'scikit-image': 'skimage',
        'scikit-learn': 'sklearn',
        'tqdm': 'tqdm',
        'requests': 'requests'
    }
    
    print("\nOptional Dependencies:")
    for package, import_name in optional_deps.items():
        try:
            importlib.import_module(import_name)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️  {package} (optional)")


def check_conda_environment():
    """Check if running in conda environment"""
    try:
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            print(f"✅ Running in conda environment: {Path(conda_prefix).name}")
            return True
        else:
            print("⚠️  Not running in conda environment")
            return False
    except Exception:
        print("⚠️  Could not determine conda environment")
        return False


def check_gpu_memory():
    """Check GPU memory if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 4:
                print("⚠️  GPU memory might be insufficient for large models")
            return True
    except Exception:
        print("⚠️  Could not check GPU memory")
        return False


def main():
    """Run all verification checks"""
    print("=== DocParser Installation Verification ===\n")
    
    # Import os for conda check
    import os
    
    checks = [
        ("Python Version", check_python_version()),
        ("PyTorch", check_pytorch()),
        ("Transformers", check_transformers()),
        ("Pillow", check_pillow()),
        ("PyMuPDF", check_pymupdf()),
        ("DocParser", check_docparser()),
    ]
    
    # Optional checks
    check_conda_environment()
    check_gpu_memory()
    check_optional_dependencies()
    
    # Summary
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    print(f"Core dependencies: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ All core checks passed! DocParser is ready to use.")
        print("\nNext steps:")
        print("1. Try the basic usage example:")
        print("   python examples/basic_usage.py")
        print("2. Check the documentation for advanced usage")
        print("3. Report any issues on GitHub")
    else:
        print("\n❌ Some checks failed. Please review the installation.")
        print("\nTroubleshooting tips:")
        print("1. Ensure you're in the correct conda environment")
        print("2. Reinstall dependencies: pip install -r requirements.txt")
        print("3. Check the README for detailed installation instructions")
        print("4. Report issues on GitHub with the full error output")


if __name__ == "__main__":
    main() 