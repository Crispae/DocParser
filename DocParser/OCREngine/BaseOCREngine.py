# Abstract Base Classes


from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from DocParser.Config.OCRConfig import OCRConfig
from DocParser.Config.ModelConfig import ModelConfig
from DocParser.Config.DataModels import OCRResult


class BaseOCREngine(ABC):
    """Abstract base class for all OCR engines"""
    
    def __init__(self, config: OCRConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the OCR engine"""
        pass
    
    @abstractmethod
    def process_image(self, image_path: Union[str, Path]) -> OCRResult:
        """Process a single image"""
        pass
    
    @abstractmethod
    def process_pdf(self, pdf_path: Union[str, Path]) -> List[OCRResult]:
        """Process a PDF document"""
        pass
    
    @abstractmethod
    def process_batch(self, file_paths: List[Union[str, Path]]) -> List[OCRResult]:
        """Process multiple files"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        pass
    
    def cleanup(self):
        """Cleanup resources"""
        pass