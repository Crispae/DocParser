#!/usr/bin/env python3
"""
Advanced usage example for DocParser with custom preprocessing and postprocessing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DocParser import DocParser, OCREngine
from DocParser.Config import OCRConfig, ModelConfig
from DocParser.Preprocessor.BasePreprocessor import BasePreprocessor
from DocParser.Postprocessor.BasePostprocessor import BasePostprocessor
from PIL import Image, ImageEnhance, ImageFilter
import re


class CustomPreprocessor(BasePreprocessor):
    """Custom preprocessor with advanced image enhancement"""
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """Apply custom preprocessing steps"""
        try:
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for optimal processing
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Apply advanced enhancement
            if self.config.preserve_layout:
                # Enhance sharpness
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.3)
                
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                
                # Enhance brightness
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.1)
                
                # Apply slight blur to reduce noise
                image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            return image
            
        except Exception as e:
            print(f"Error in custom preprocessing: {e}")
            return image


class CustomPostprocessor(BasePostprocessor):
    """Custom postprocessor with advanced text cleaning"""
    
    def postprocess(self, result):
        """Apply custom postprocessing steps"""
        try:
            # Filter by confidence threshold
            if self.config.confidence_threshold > 0:
                filtered_elements = [
                    element for element in result.elements 
                    if element.confidence >= self.config.confidence_threshold
                ]
                result.elements = filtered_elements
                result.text = " ".join([elem.text for elem in filtered_elements])
            
            # Advanced text cleaning
            if result.text:
                result.text = self._advanced_clean_text(result.text)
                for element in result.elements:
                    element.text = self._advanced_clean_text(element.text)
            
            # Extract structured information
            result = self._extract_structured_info(result)
            
            return result
            
        except Exception as e:
            print(f"Error in custom postprocessing: {e}")
            return result
    
    def _advanced_clean_text(self, text: str) -> str:
        """Advanced text cleaning with multiple steps"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        
        # Fix common OCR mistakes
        text = text.replace('0', 'o')  # Common OCR mistake
        text = text.replace('1', 'l')  # Common OCR mistake
        text = text.replace('5', 's')  # Common OCR mistake
        
        # Remove duplicate characters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return text.strip()
    
    def _extract_structured_info(self, result):
        """Extract structured information from text"""
        # This is a placeholder for structured information extraction
        # In a real implementation, you might extract:
        # - Email addresses
        # - Phone numbers
        # - Dates
        # - Names
        # - Addresses
        
        # Example: Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, result.text)
        
        if emails:
            result.metadata['emails'] = emails
        
        return result


def main():
    """Demonstrate advanced DocParser usage"""
    
    print("=== DocParser Advanced Usage Example ===\n")
    
    # Create configurations
    config = OCRConfig(
        confidence_threshold=0.6,
        preserve_layout=True,
        extract_tables=True
    )
    
    model_config = ModelConfig(
        device="cpu",
        batch_size=1
    )
    
    # Example file path
    image_path = "example_document.png"
    
    print("1. Processing with Custom Preprocessing and Postprocessing:")
    
    try:
        # Create custom preprocessor and postprocessor
        custom_preprocessor = CustomPreprocessor(config)
        custom_postprocessor = CustomPostprocessor(config)
        
        # Process with custom components
        with DocParser(OCREngine.NANONETS, config, model_config) as parser:
            # Override preprocessor and postprocessor
            parser.engine.preprocessor = custom_preprocessor
            parser.engine.postprocessor = custom_postprocessor
            
            if os.path.exists(image_path):
                result = parser.process_image(image_path)
                
                print(f"   Original text length: {len(result.text)}")
                print(f"   Text preview: {result.text[:200]}...")
                print(f"   Processing time: {result.processing_time:.2f}s")
                print(f"   Elements found: {len(result.elements)}")
                
                # Show extracted metadata
                if 'emails' in result.metadata:
                    print(f"   Emails found: {result.metadata['emails']}")
                
            else:
                print(f"   Image file not found: {image_path}")
                
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. Batch Processing Example:")
    
    # Example batch processing
    files = ["doc1.png", "doc2.jpg", "doc3.pdf"]
    existing_files = [f for f in files if os.path.exists(f)]
    
    if existing_files:
        try:
            with DocParser(OCREngine.SMALL_DOCLING, config, model_config) as parser:
                results = parser.process_batch(existing_files)
                
                for file_path, result in zip(existing_files, results):
                    print(f"   {file_path}: {len(result.text)} characters")
                    
        except Exception as e:
            print(f"   Error in batch processing: {e}")
    else:
        print("   No example files found for batch processing")
    
    print("\n3. Performance Comparison:")
    compare_performance(image_path)


def compare_performance(image_path):
    """Compare performance of different engines"""
    if not os.path.exists(image_path):
        print("   Image file not found for performance comparison")
        return
    
    engines = [OCREngine.NANONETS, OCREngine.SMALL_DOCLING]
    config = OCRConfig(confidence_threshold=0.5)
    model_config = ModelConfig(device="cpu")
    
    print("   Engine | Time (s) | Text Length | Confidence")
    print("   ------|----------|-------------|------------")
    
    for engine_type in engines:
        try:
            with DocParser(engine_type, config, model_config) as parser:
                result = parser.process_image(image_path)
                
                print(f"   {engine_type.value:6} | {result.processing_time:8.2f} | "
                      f"{len(result.text):11} | {result.confidence_score:10.2f}")
                
        except Exception as e:
            print(f"   {engine_type.value:6} | {'ERROR':>8} | {'N/A':>11} | {'N/A':>10}")


if __name__ == "__main__":
    main() 