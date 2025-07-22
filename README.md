# DocParser - Multi-Engine VLM-Based OCR Framework

A unified interface for Vision Language Model (VLM) based OCR engines, providing seamless switching between different state-of-the-art document understanding models.

## Features

- **Multi-Engine Support**: Switch between different VLM-based OCR engines
- **Unified Interface**: Consistent API across all engines
- **Preprocessing & Postprocessing**: Engine-specific image enhancement and result refinement
- **Batch Processing**: Efficient processing of multiple documents
- **GPU/CPU Support**: Optimized for both CPU and GPU environments
- **Resource Management**: Automatic cleanup and memory management

## Supported OCR Engines

| Engine | Model | Architecture | Features |
|--------|-------|--------------|----------|
| **Nanonets** | Nanonets-OCR-s | Vision Encoder-Decoder | High accuracy, fast processing |
| **Dolphin** | ByteDance Dolphin | Analyze-then-Parse VLM | Advanced document understanding |
| **OlmOCR** | OlmOCR-7B | Large VLM | Complex document structure |
| **SmallDocling** | SmolDocling-256M | Lightweight VLM | Efficient, DocTags output |

## System Requirements

### Python Version
- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **CUDA 11.8+** (for GPU support)
- **8GB+ RAM** (16GB+ recommended for large models)

### Operating System
- **Linux** (Ubuntu 18.04+ recommended)
- **Windows 10/11** (with WSL2 recommended)
- **macOS** (10.15+)

## Installation

### Option 1: Conda Environment (Recommended)

#### Create Conda Environment
```bash
# Create a new conda environment with Python 3.9
conda create -n docparser python=3.9

# Activate the environment
conda activate docparser
```

#### Install PyTorch with CUDA Support
```bash
# For CUDA 11.8 (adjust version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU-only installation
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### Install Core Dependencies
```bash
# Install core dependencies
conda install -c conda-forge pillow numpy scikit-learn tqdm

# Install additional dependencies
pip install transformers>=4.30.0
pip install PyMuPDF>=1.23.0
pip install pytesseract>=0.3.10
pip install opencv-python>=4.8.0
pip install scikit-image>=0.21.0
```

#### Install Optional Dependencies
```bash
# For OlmOCR engine
pip install olmocr[gpu]

# For additional OCR engines
pip install easyocr

# For development
pip install pytest black flake8
```

### Option 2: Pip Installation

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv docparser_env

# Activate environment (Linux/macOS)
source docparser_env/bin/activate

# Activate environment (Windows)
docparser_env\Scripts\activate
```

#### Install Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
```

### Option 3: Docker (Coming Soon)

```bash
# Pull the Docker image
docker pull docparser/docparser:latest

# Run with GPU support
docker run --gpus all -v $(pwd):/workspace docparser/docparser:latest
```

## Environment Setup Verification

### Check Python Version
```bash
python --version
# Should show Python 3.8+
```

### Check PyTorch Installation
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
```

### Test Installation
```python
from DocParser import DocParser, OCREngine
print("DocParser installation successful!")
```

## Quick Start

### Basic Usage

```python
from DocParser import DocParser, OCREngine
from DocParser.Config import OCRConfig, ModelConfig

# Create configurations
config = OCRConfig(
    confidence_threshold=0.7,
    preserve_layout=True,
    extract_tables=True
)

model_config = ModelConfig(
    device="cuda",  # or "cpu"
    batch_size=1
)

# Initialize parser with Nanonets engine
with DocParser(OCREngine.NANONETS, config, model_config) as parser:
    # Process single image
    result = parser.process_image("document.png")
    print(f"Extracted text: {result.text}")
    
    # Process PDF
    results = parser.process_pdf("document.pdf")
    for i, page_result in enumerate(results):
        print(f"Page {i+1}: {page_result.text}")
```

### Convenience Functions

```python
from DocParser import process_image, process_pdf, OCREngine

# Quick image processing
result = process_image("document.png", OCREngine.DOLPHIN)

# Quick PDF processing
results = process_pdf("document.pdf", OCREngine.OLMOCR)
```

### Batch Processing

```python
from DocParser import DocParser, OCREngine

files = ["doc1.pdf", "doc2.png", "doc3.jpg"]

with DocParser(OCREngine.SMALL_DOCLING) as parser:
    results = parser.process_batch(files)
    
    for file_path, result in zip(files, results):
        print(f"{file_path}: {result.text[:100]}...")
```

## Configuration

### OCR Configuration

```python
from DocParser.Config import OCRConfig

config = OCRConfig(
    confidence_threshold=0.7,    # Minimum confidence for text elements
    preserve_layout=True,         # Maintain document layout
    extract_tables=True,          # Extract table structures
    extract_images=False,         # Extract embedded images
    language="en",               # Language for OCR
    dpi=300                     # Resolution for processing
)
```

### Model Configuration

```python
from DocParser.Config import ModelConfig

model_config = ModelConfig(
    device="cuda",               # "cuda" or "cpu"
    batch_size=4,               # Batch size for processing
    model_path=None,            # Custom model path
    api_key=None,               # API key for cloud services
    api_url=None,               # API endpoint
    custom_params={}            # Engine-specific parameters
)
```

## Engine-Specific Features

### Nanonets Engine
- **Best for**: General text recognition
- **Strengths**: Fast, accurate, good for clean documents
- **Limitations**: No bounding box information

### Dolphin Engine
- **Best for**: Complex document understanding
- **Strengths**: Analyze-then-parse approach, table extraction
- **Features**: Advanced document structure analysis

### OlmOCR Engine
- **Best for**: Large, complex documents
- **Strengths**: 7B parameter model, excellent accuracy
- **Requirements**: Requires olmocr package installation

### SmallDocling Engine
- **Best for**: Lightweight processing
- **Strengths**: Efficient, DocTags structured output
- **Features**: Document structure parsing

## Advanced Usage

### Custom Preprocessing

```python
from DocParser.Preprocessor import BasePreprocessor
from PIL import Image

class CustomPreprocessor(BasePreprocessor):
    def preprocess(self, image: Image.Image) -> Image.Image:
        # Custom preprocessing logic
        image = image.convert('RGB')
        # Add your custom preprocessing steps
        return image
```

### Custom Postprocessing

```python
from DocParser.Postprocessor import BasePostprocessor
from DocParser.Config.DataModels import OCRResult

class CustomPostprocessor(BasePostprocessor):
    def postprocess(self, result: OCRResult) -> OCRResult:
        # Custom postprocessing logic
        # Clean text, filter results, etc.
        return result
```

### Engine Comparison

```python
from DocParser import DocParser, OCREngine
import time

def compare_engines(image_path):
    engines = [OCREngine.NANONETS, OCREngine.DOLPHIN, OCREngine.SMALL_DOCLING]
    
    for engine_type in engines:
        start_time = time.time()
        
        with DocParser(engine_type) as parser:
            result = parser.process_image(image_path)
        
        processing_time = time.time() - start_time
        
        print(f"{engine_type.value}:")
        print(f"  Text: {result.text[:100]}...")
        print(f"  Time: {processing_time:.2f}s")
        print(f"  Confidence: {result.confidence_score}")
        print()
```

## Performance Optimization

### GPU Usage
```python
model_config = ModelConfig(device="cuda")
```

### Batch Processing
```python
model_config = ModelConfig(batch_size=8)
```

### Memory Management
```python
# Automatic cleanup with context manager
with DocParser(OCREngine.NANONETS) as parser:
    result = parser.process_image("document.png")

# Manual cleanup
parser = DocParser(OCREngine.NANONETS)
try:
    result = parser.process_image("document.png")
finally:
    parser.cleanup()
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size
model_config = ModelConfig(batch_size=1)

# Use CPU instead
model_config = ModelConfig(device="cpu")
```

#### Model Download Issues
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Set environment variable for offline mode
export HF_DATASETS_OFFLINE=1
```

#### Conda Environment Issues
```bash
# Update conda
conda update conda

# Clean conda cache
conda clean --all

# Recreate environment
conda env remove -n docparser
conda create -n docparser python=3.9
```

### Environment Verification Script

Create a file `verify_installation.py`:
```python
#!/usr/bin/env python3
"""Verify DocParser installation"""

import sys
import torch

def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_pytorch():
    try:
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def check_docparser():
    try:
        from DocParser import DocParser, OCREngine
        print("✅ DocParser imported successfully")
        return True
    except Exception as e:
        print(f"❌ DocParser import error: {e}")
        return False

if __name__ == "__main__":
    print("=== DocParser Installation Verification ===\n")
    
    checks = [
        check_python_version(),
        check_pytorch(),
        check_docparser()
    ]
    
    if all(checks):
        print("\n✅ All checks passed! DocParser is ready to use.")
    else:
        print("\n❌ Some checks failed. Please review the installation.")
```

Run the verification:
```bash
python verify_installation.py
```

## Error Handling

```python
from DocParser import DocParser, OCREngine

try:
    with DocParser(OCREngine.NANONETS) as parser:
        result = parser.process_image("document.png")
except RuntimeError as e:
    print(f"Engine initialization failed: {e}")
except Exception as e:
    print(f"Processing failed: {e}")
```

## Supported File Formats

- **PDF**: Multi-page documents
- **Images**: JPG, PNG, BMP, TIFF, GIF
- **Scanned Documents**: All image formats
- **Handwritten Text**: Supported by VLM engines

## Architecture

```
DocParser/
├── Config/
│   ├── DataModels.py      # Data structures
│   ├── OCRConfig.py       # OCR configuration
│   └── ModelConfig.py     # Model configuration
├── OCREngine/
│   ├── BaseOCREngine.py   # Abstract base class
│   ├── nanonetsocr.py     # Nanonets implementation
│   ├── dolphinocr.py      # Dolphin implementation
│   ├── olmocr.py          # OlmOCR implementation
│   └── SmallDocling.py    # SmallDocling implementation
├── Preprocessor/
│   └── BasePreprocessor.py # Image preprocessing
└── Postprocessor/
    └── BasePostprocessor.py # Result postprocessing
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{docparser2024,
  title={DocParser: Multi-Engine VLM-Based OCR Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/docparser}
}
```
