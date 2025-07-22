# Configuration Classes

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class OCRConfig:
    """Base configuration for OCR operations"""
    confidence_threshold: float = 0.5
    preserve_layout: bool = True
    extract_tables: bool = False
    extract_images: bool = False
    language: str = "en"
    dpi: int = 300
