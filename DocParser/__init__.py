from typing import Union, List, Optional
from pathlib import Path
import logging

from DocParser.Config.OCRConfig import OCRConfig
from DocParser.Config.ModelConfig import ModelConfig
from DocParser.Config.DataModels import OCREngine as OCREngineEnum, OCRResult as OCRResultType
from DocParser.OCREngine import create_ocr_engine, OCREngineFactory


class DocParser:
    """Main OCR processor class providing unified interface for multiple OCR engines"""
    
    def __init__(self, engine_type: Union[str, OCREngineEnum] = OCREngineEnum.NANONETS, 
                 config: Optional[OCRConfig] = None, 
                 model_config: Optional[ModelConfig] = None):
        """
        Initialize DocParser with specified OCR engine
        
        Args:
            engine_type: Type of OCR engine to use
            config: OCR configuration
            model_config: Model-specific configuration
        """
        self.engine_type = engine_type if isinstance(engine_type, OCREngineEnum) else OCREngineEnum(engine_type)
        
        # Set default configurations if not provided
        self.config = config or OCRConfig()
        self.model_config = model_config or ModelConfig()
        
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create and initialize OCR engine
        self.engine = create_ocr_engine(self.engine_type, self.config, self.model_config)
        
        # Initialize the engine
        if not self.engine.initialize():
            raise RuntimeError(f"Failed to initialize {self.engine_type.value} OCR engine")
        
        self.logger.info(f"Initialized {self.engine_type.value} OCR engine successfully")
    
    def process_image(self, image_path: Union[str, Path]) -> OCRResultType:
        """Process a single image"""
        return self.engine.process_image(image_path)
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> List[OCRResultType]:
        """Process a PDF document"""
        return self.engine.process_pdf(pdf_path)
    
    def process_batch(self, file_paths: List[Union[str, Path]]) -> List[OCRResultType]:
        """Process multiple files"""
        return self.engine.process_batch(file_paths)
    
    def get_supported_formats(self) -> List[str]:
        """Get supported file formats for current engine"""
        return self.engine.get_supported_formats()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'engine'):
            self.engine.cleanup()
    
    @classmethod
    def get_available_engines(cls) -> List[OCREngineEnum]:
        """Get list of available OCR engines"""
        return OCREngineFactory.get_available_engines()
    
    @classmethod
    def get_engine_info(cls, engine_type: OCREngineEnum) -> dict:
        """Get information about a specific engine"""
        return OCREngineFactory.get_engine_info(engine_type)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


# Convenience functions
def process_image(image_path: Union[str, Path], 
                 engine_type: Union[str, OCREngineEnum] = OCREngineEnum.NANONETS,
                 config: Optional[OCRConfig] = None,
                 model_config: Optional[ModelConfig] = None) -> OCRResultType:
    """Process a single image with specified engine"""
    with DocParser(engine_type, config, model_config) as parser:
        return parser.process_image(image_path)


def process_pdf(pdf_path: Union[str, Path],
                engine_type: Union[str, OCREngineEnum] = OCREngineEnum.NANONETS,
                config: Optional[OCRConfig] = None,
                model_config: Optional[ModelConfig] = None) -> List[OCRResultType]:
    """Process a PDF document with specified engine"""
    with DocParser(engine_type, config, model_config) as parser:
        return parser.process_pdf(pdf_path)


def process_batch(file_paths: List[Union[str, Path]],
                 engine_type: Union[str, OCREngineEnum] = OCREngineEnum.NANONETS,
                 config: Optional[OCRConfig] = None,
                 model_config: Optional[ModelConfig] = None) -> List[OCRResultType]:
    """Process multiple files with specified engine"""
    with DocParser(engine_type, config, model_config) as parser:
        return parser.process_batch(file_paths)

# Expose OCREngine and OCRResult at the package level for convenience
OCREngine = OCREngineEnum
OCRResult = OCRResultType
