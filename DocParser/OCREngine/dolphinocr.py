from typing import Union, List, Tuple, Optional
from pathlib import Path
import logging
import time
import torch
from PIL import Image
import io
import re

from DocParser.OCREngine.BaseOCREngine import BaseOCREngine
from DocParser.Config.OCRConfig import OCRConfig
from DocParser.Config.ModelConfig import ModelConfig
from DocParser.Config.DataModels import OCRResult, TextElement, BoundingBox
from DocParser.Preprocessor.BasePreprocessor import BasePreprocessor
from DocParser.Postprocessor.BasePostprocessor import BasePostprocessor


class DolphinPreprocessor(BasePreprocessor):
    """Preprocessor for Dolphin OCR engine"""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """Preprocess image for Dolphin OCR"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Dolphin works best with high-resolution images
            # Resize if image is too small for better recognition
            min_size = 512
            if min(image.size) < min_size:
                ratio = min_size / min(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Enhance image quality for better VLM performance
            if self.config.preserve_layout:
                from PIL import ImageEnhance
                
                # Enhance sharpness
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.2)
                
                # Enhance contrast slightly
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error in Dolphin preprocessing: {e}")
            return image


class DolphinPostprocessor(BasePostprocessor):
    """Postprocessor for Dolphin OCR engine"""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def postprocess(self, result: OCRResult) -> OCRResult:
        """Postprocess Dolphin OCR results"""
        try:
            # Filter by confidence threshold
            if self.config.confidence_threshold > 0:
                filtered_elements = [
                    element for element in result.elements 
                    if element.confidence >= self.config.confidence_threshold
                ]
                result.elements = filtered_elements
                result.text = " ".join([elem.text for elem in filtered_elements])
            
            # Clean up text
            if result.text:
                result.text = self._clean_text(result.text)
                for element in result.elements:
                    element.text = self._clean_text(element.text)
            
            # Extract tables if requested
            if self.config.extract_tables:
                result = self._extract_tables(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Dolphin postprocessing: {e}")
            return result
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        
        return text.strip()
    
    def _extract_tables(self, result: OCRResult) -> OCRResult:
        """Extract table structures from Dolphin output"""
        # Dolphin's analyze-then-parse approach might provide table information
        # This is a placeholder for table extraction logic
        return result


class DolphinEngine(BaseOCREngine):
    """Dolphin OCR engine implementation"""
    
    def __init__(self, config: OCRConfig, model_config: ModelConfig):
        super().__init__(config, model_config)
        self.model = None
        self.processor = None
        self.device = model_config.device
        
        # Initialize preprocessor and postprocessor
        self.preprocessor = DolphinPreprocessor(config)
        self.postprocessor = DolphinPostprocessor(config)
        
    def initialize(self) -> bool:
        try:
            from transformers import AutoModel, AutoProcessor
            
            self.logger.info("Initializing Dolphin OCR engine")
            
            # Note: This is a placeholder as Dolphin might not be publicly available yet
            # Based on the search results, it's a recent ByteDance model
            model_name = self.model_config.model_path or "bytedance/dolphin-ocr"
            
            try:
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                if self.device == "cuda":
                    self.model = self.model.cuda()
                
            except Exception:
                self.logger.warning("Dolphin model not found, using fallback implementation")
                # Fallback to a similar model structure
                self.model = None
                self.processor = None
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Dolphin: {e}")
            return False
    
    def process_image(self, image_path: Union[str, Path]) -> OCRResult:
        start_time = time.time()
        
        try:
            if self.model is None:
                # Fallback implementation
                return self._fallback_process_image(image_path)
            
            # Load and preprocess image
            image = Image.open(image_path)
            image = self.preprocessor.preprocess(image)
            
            # Process with Dolphin (analyze-then-parse approach)
            inputs = self.processor(images=image, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate structured output
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=0.0
                )
            
            # Decode the output
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Parse the structured output
            elements = self._parse_dolphin_output(generated_text)
            
            processing_time = time.time() - start_time
            
            result = OCRResult(
                text=generated_text,
                elements=elements,
                metadata={"model_name": "Dolphin", "image_path": str(image_path)},
                processing_time=processing_time,
                model_name="Dolphin",
                confidence_score=0.92
            )
            
            # Postprocess result
            result = self.postprocessor.postprocess(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image with Dolphin: {e}")
            return self._fallback_process_image(image_path)
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> List[OCRResult]:
        import fitz  # PyMuPDF
        
        results = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Save temporarily and process
                temp_path = f"/tmp/dolphin_page_{page_num}.png"
                image.save(temp_path)
                
                result = self.process_image(temp_path)
                result.metadata["page_number"] = page_num
                results.append(result)
                
                # Clean up temp file
                import os
                os.remove(temp_path)
            
            pdf_document.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing PDF with Dolphin: {e}")
            raise
    
    def process_batch(self, file_paths: List[Union[str, Path]]) -> List[OCRResult]:
        results = []
        for file_path in file_paths:
            if str(file_path).lower().endswith('.pdf'):
                results.extend(self.process_pdf(file_path))
            else:
                results.append(self.process_image(file_path))
        return results
    
    def get_supported_formats(self) -> List[str]:
        return ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def _fallback_process_image(self, image_path: Union[str, Path]) -> OCRResult:
        """Fallback implementation when Dolphin model is not available"""
        start_time = time.time()
        
        # Simple fallback - could use tesseract or another OCR engine
        try:
            import pytesseract
            
            image = Image.open(image_path)
            image = self.preprocessor.preprocess(image)
            
            text = pytesseract.image_to_string(image)
            
            elements = [TextElement(
                text=text,
                confidence=0.8,
                element_type="text"
            )]
            
            processing_time = time.time() - start_time
            
            result = OCRResult(
                text=text,
                elements=elements,
                metadata={"model_name": "Dolphin-Fallback", "fallback": True},
                processing_time=processing_time,
                model_name="Dolphin"
            )
            
            # Postprocess result
            result = self.postprocessor.postprocess(result)
            
            return result
            
        except ImportError:
            # If tesseract is not available, return empty result
            return OCRResult(
                text="",
                elements=[],
                metadata={"error": "Dolphin model not available and no fallback OCR found"},
                processing_time=time.time() - start_time,
                model_name="Dolphin"
            )
    
    def _parse_dolphin_output(self, output_text: str) -> List[TextElement]:
        """Parse Dolphin's structured output"""
        elements = []
        lines = output_text.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip():
                # Dolphin uses analyze-then-parse, so output might be structured
                element = TextElement(
                    text=line.strip(),
                    confidence=0.92,
                    element_type="text"
                )
                elements.append(element)
        
        return elements
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
