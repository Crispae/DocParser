from typing import Union, List, Tuple, Optional
from pathlib import Path
import logging
import time
import json
import subprocess
import os
import glob
import re

from DocParser.OCREngine.BaseOCREngine import BaseOCREngine
from DocParser.Config.OCRConfig import OCRConfig
from DocParser.Config.ModelConfig import ModelConfig
from DocParser.Config.DataModels import OCRResult, TextElement, BoundingBox
from DocParser.Preprocessor.BasePreprocessor import BasePreprocessor
from DocParser.Postprocessor.BasePostprocessor import BasePostprocessor


class OlmocrPreprocessor(BasePreprocessor):
    """Preprocessor for OlmOCR engine"""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def preprocess(self, image) -> str:
        """Preprocess image for OlmOCR - returns file path since OlmOCR works with files"""
        try:
            # OlmOCR expects file paths, so we return the path as-is
            # The actual preprocessing happens within OlmOCR pipeline
            return str(image) if isinstance(image, (str, Path)) else image
            
        except Exception as e:
            self.logger.error(f"Error in OlmOCR preprocessing: {e}")
            return str(image) if isinstance(image, (str, Path)) else image


class OlmocrPostprocessor(BasePostprocessor):
    """Postprocessor for OlmOCR engine"""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def postprocess(self, result: OCRResult) -> OCRResult:
        """Postprocess OlmOCR results"""
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in OlmOCR postprocessing: {e}")
            return result
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        
        return text.strip()


class OlmocrEngine(BaseOCREngine):
    """OlmOCR engine implementation"""
    
    def __init__(self, config: OCRConfig, model_config: ModelConfig):
        super().__init__(config, model_config)
        self.model_path = model_config.model_path or "allenai/olmOCR-7B-0225-preview"
        self.workspace_dir = "/tmp/olmocr_workspace"
        
        # Initialize preprocessor and postprocessor
        self.preprocessor = OlmocrPreprocessor(config)
        self.postprocessor = OlmocrPostprocessor(config)
        
    def initialize(self) -> bool:
        try:
            self.logger.info("Initializing OlmOCR engine")
            
            # Create workspace directory
            os.makedirs(self.workspace_dir, exist_ok=True)
            
            # Check if olmocr is installed
            result = subprocess.run(["python", "-m", "olmocr.pipeline", "--help"], 
                                 capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error("OlmOCR not installed. Please install with: pip install olmocr[gpu]")
                return False
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OlmOCR: {e}")
            return False
    
    def process_image(self, image_path: Union[str, Path]) -> OCRResult:
        start_time = time.time()
        
        try:
            # Preprocess image path
            processed_path = self.preprocessor.preprocess(image_path)
            
            # Run OlmOCR pipeline
            cmd = [
                "python", "-m", "olmocr.pipeline", 
                self.workspace_dir,
                "--pdfs", str(processed_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_dir)
            
            if result.returncode != 0:
                raise Exception(f"OlmOCR failed: {result.stderr}")
            
            # Read results
            result_files = glob.glob(f"{self.workspace_dir}/results/output_*.jsonl")
            
            if not result_files:
                raise Exception("No output files generated by OlmOCR")
            
            # Parse the first result
            with open(result_files[0], 'r') as f:
                line = f.readline().strip()
                if line:
                    data = json.loads(line)
                    text = data.get('text', '')
                    
                    # Create TextElements from the extracted text
                    elements = [TextElement(
                        text=text,
                        confidence=0.9,  # OlmOCR doesn't provide confidence scores
                        element_type="document",
                        page_number=0
                    )]
                    
                    processing_time = time.time() - start_time
                    
                    result = OCRResult(
                        text=text,
                        elements=elements,
                        metadata={"model_name": "olmOCR-7B", "source_file": str(image_path)},
                        processing_time=processing_time,
                        model_name="OlmOCR"
                    )
                    
                    # Postprocess result
                    result = self.postprocessor.postprocess(result)
                    
                    return result
            
            raise Exception("No valid results found in OlmOCR output")
            
        except Exception as e:
            self.logger.error(f"Error processing image with OlmOCR: {e}")
            raise
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> List[OCRResult]:
        start_time = time.time()
        
        try:
            # Preprocess PDF path
            processed_path = self.preprocessor.preprocess(pdf_path)
            
            # Run OlmOCR pipeline for PDF
            cmd = [
                "python", "-m", "olmocr.pipeline", 
                self.workspace_dir,
                "--pdfs", str(processed_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_dir)
            
            if result.returncode != 0:
                raise Exception(f"OlmOCR failed: {result.stderr}")
            
            # Read results
            result_files = glob.glob(f"{self.workspace_dir}/results/output_*.jsonl")
            
            if not result_files:
                raise Exception("No output files generated by OlmOCR")
            
            results = []
            
            # Parse all results (OlmOCR can output multiple pages)
            for result_file in result_files:
                with open(result_file, 'r') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            text = data.get('text', '')
                            
                            elements = [TextElement(
                                text=text,
                                confidence=0.9,
                                element_type="document",
                                page_number=line_num
                            )]
                            
                            processing_time = time.time() - start_time
                            
                            result = OCRResult(
                                text=text,
                                elements=elements,
                                metadata={"model_name": "olmOCR-7B", "source_file": str(pdf_path), "page": line_num},
                                processing_time=processing_time,
                                model_name="OlmOCR"
                            )
                            
                            # Postprocess result
                            result = self.postprocessor.postprocess(result)
                            results.append(result)
            
            return results if results else [OCRResult("", [], {}, 0.0, "OlmOCR")]
            
        except Exception as e:
            self.logger.error(f"Error processing PDF with OlmOCR: {e}")
            raise
    
    def process_batch(self, file_paths: List[Union[str, Path]]) -> List[OCRResult]:
        try:
            # Preprocess all file paths
            processed_paths = [self.preprocessor.preprocess(path) for path in file_paths]
            
            # Run OlmOCR pipeline for batch processing
            file_paths_str = " ".join([str(p) for p in processed_paths])
            
            cmd = [
                "python", "-m", "olmocr.pipeline", 
                self.workspace_dir,
                "--pdfs", file_paths_str
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_dir)
            
            if result.returncode != 0:
                raise Exception(f"OlmOCR batch processing failed: {result.stderr}")
            
            # Read all results
            result_files = glob.glob(f"{self.workspace_dir}/results/output_*.jsonl")
            results = []
            
            for result_file in result_files:
                with open(result_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            text = data.get('text', '')
                            
                            elements = [TextElement(
                                text=text,
                                confidence=0.9,
                                element_type="document"
                            )]
                            
                            result = OCRResult(
                                text=text,
                                elements=elements,
                                metadata={"model_name": "olmOCR-7B"},
                                processing_time=0.0,
                                model_name="OlmOCR"
                            )
                            
                            # Postprocess result
                            result = self.postprocessor.postprocess(result)
                            results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch processing with OlmOCR: {e}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        return ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Clean up workspace directory
            import shutil
            if os.path.exists(self.workspace_dir):
                shutil.rmtree(self.workspace_dir)
        except Exception as e:
            self.logger.error(f"Error cleaning up OlmOCR workspace: {e}")