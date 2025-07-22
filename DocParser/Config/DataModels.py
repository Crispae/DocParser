from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

@dataclass
class BoundingBox:
    """ Represents a bounding box with coordinates. """
    x: float
    y: float
    width: float
    height: float


@dataclass
class TextElement:
    """ Represents extracted with metadata """
    text: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    element_type: Optional[str] = None
    page_number: Optional[int] = None


@dataclass
class OCRResult:
    """Complete OCR result containing all extracted information"""
    text: str
    elements: List[TextElement]
    metadata: Dict[str, Any]
    processing_time: float
    model_name: str
    confidence_score: Optional[float] = None


class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    IMAGE = "image"
    SCAN = "scan"
    HANDWRITTEN = "handwritten"
    FORM = "form"
    TABLE = "table"

class OCREngine(Enum):
    """Supported OCR engines"""
    DOCLING = "docling"
    OLMOCR = "olmocr"
    SMALL_DOCLING = "small_docling"
    NANONETS = "nanonets"
    DOLPHIN = "dolphin"
    NOGGUT = "noggut"
    MARKITDOWN = "markitdown"

