#!/usr/bin/env python3
"""
Basic usage example for DocParser
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DocParser import DocParser, OCREngine
from DocParser.Config import OCRConfig, ModelConfig


def main():
    """Demonstrate basic DocParser usage"""
    
    # Create configurations
    config = OCRConfig(
        confidence_threshold=0.7,
        preserve_layout=True,
        extract_tables=True
    )
    
    model_config = ModelConfig(
        device="cpu",  # Use "cuda" if GPU is available
        batch_size=1
    )
    
    # Example file paths (replace with your actual files)
    image_path = "example_document.png"
    pdf_path = "example_document.pdf"
    
    print("=== DocParser Basic Usage Example ===\n")
    
    # Test with Nanonets engine
    print("1. Processing with Nanonets Engine:")
    try:
        with DocParser(OCREngine.NANONETS, config, model_config) as parser:
            if os.path.exists(image_path):
                result = parser.process_image(image_path)
                print(f"   Extracted text: {result.text[:200]}...")
                print(f"   Processing time: {result.processing_time:.2f}s")
                print(f"   Elements found: {len(result.elements)}")
            else:
                print(f"   Image file not found: {image_path}")
    except Exception as e:
        print(f"   Error with Nanonets: {e}")
    
    print("\n2. Available Engines:")
    engines = DocParser.get_available_engines()
    for engine in engines:
        info = DocParser.get_engine_info(engine)
        print(f"   - {info['name']}: {info['class']}")
    
    print("\n3. Engine Comparison:")
    if os.path.exists(image_path):
        compare_engines(image_path)
    else:
        print(f"   Image file not found: {image_path}")


def compare_engines(image_path):
    """Compare different OCR engines"""
    engines = [OCREngine.NANONETS, OCREngine.SMALL_DOCLING]
    
    config = OCRConfig(confidence_threshold=0.5)
    model_config = ModelConfig(device="cpu")
    
    for engine_type in engines:
        try:
            print(f"   Testing {engine_type.value}...")
            
            with DocParser(engine_type, config, model_config) as parser:
                result = parser.process_image(image_path)
                
                print(f"     Text: {result.text[:100]}...")
                print(f"     Time: {result.processing_time:.2f}s")
                print(f"     Confidence: {result.confidence_score}")
                
        except Exception as e:
            print(f"     Error: {e}")
        
        print()


if __name__ == "__main__":
    main() 