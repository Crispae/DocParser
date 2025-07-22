from abc import ABC,abstractmethod
from DocParser.Config.DataModels import OCRResult

class BasePostprocessor(ABC):
    """Abstract base class for result postprocessing"""
    
    @abstractmethod
    def postprocess(self, result: OCRResult) -> OCRResult:
        """Postprocess the OCR result"""
        pass