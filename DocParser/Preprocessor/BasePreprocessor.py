from abc import ABC, abstractmethod
from typing import Any, Union
from PIL import Image
from DocParser.Config.OCRConfig import OCRConfig
from DocParser.Config.DataModels import OCRResult


class BasePreprocessor(ABC):
    """Abstract base class for image preprocessing"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
    
    @abstractmethod
    def preprocess(self, image: Image.Image) -> Image.Image:
        """Preprocess the image"""
        pass


class BasePostprocessor(ABC):
    """Abstract base class for result postprocessing"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
    
    @abstractmethod
    def postprocess(self, result: OCRResult) -> OCRResult:
        """Postprocess the OCR result"""
        pass