from typing import Union, Optional
from DocParser.Config.OCRConfig import OCRConfig
from DocParser.Config.ModelConfig import ModelConfig
from DocParser.Config.DataModels import OCREngine
from DocParser.OCREngine.BaseOCREngine import BaseOCREngine

# Import all OCR engines
from DocParser.OCREngine.nanonetsocr import NanonetsEngine
from DocParser.OCREngine.dolphinocr import DolphinEngine
from DocParser.OCREngine.olmocr import OlmocrEngine
from DocParser.OCREngine.SmallDocling import SmallDoclingEngine


class OCREngineFactory:
    """Factory class for creating OCR engines"""
    
    _engines = {
        OCREngine.NANONETS: NanonetsEngine,
        OCREngine.DOLPHIN: DolphinEngine,
        OCREngine.OLMOCR: OlmocrEngine,
        OCREngine.SMALL_DOCLING: SmallDoclingEngine,
    }
    
    @classmethod
    def create_engine(cls, engine_type: OCREngine, config: OCRConfig, model_config: ModelConfig) -> BaseOCREngine:
        """Create an OCR engine instance"""
        if engine_type not in cls._engines:
            raise ValueError(f"Unsupported OCR engine: {engine_type}")
        
        engine_class = cls._engines[engine_type]
        return engine_class(config, model_config)
    
    @classmethod
    def get_available_engines(cls) -> list:
        """Get list of available OCR engines"""
        return list(cls._engines.keys())
    
    @classmethod
    def get_engine_info(cls, engine_type: OCREngine) -> dict:
        """Get information about a specific engine"""
        if engine_type not in cls._engines:
            return {"error": f"Engine {engine_type} not found"}
        
        engine_class = cls._engines[engine_type]
        return {
            "name": engine_type.value,
            "class": engine_class.__name__,
            "supported_formats": engine_class.get_supported_formats() if hasattr(engine_class, 'get_supported_formats') else []
        }


def create_ocr_engine(engine_type: Union[str, OCREngine], config: OCRConfig, model_config: ModelConfig) -> BaseOCREngine:
    """Convenience function to create an OCR engine"""
    if isinstance(engine_type, str):
        try:
            engine_type = OCREngine(engine_type)
        except ValueError:
            raise ValueError(f"Invalid engine type: {engine_type}")
    
    return OCREngineFactory.create_engine(engine_type, config, model_config)
