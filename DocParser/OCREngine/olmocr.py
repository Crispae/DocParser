from typing import Union, List, Tuple, Optional
from pathlib import Path
import logging
import time
import json
import subprocess
import os
import glob
import re
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
import torch

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
        self.config = config
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
        # super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def postprocess(self, result: OCRResult) -> OCRResult:
        """Postprocess OlmOCR results"""
        try:
            # Filter by confidence threshold
            if self.config.confidence_threshold > 0:
                filtered_elements = [
                    element
                    for element in result.elements
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
        text = re.sub(r"\s+", " ", text)

        # Remove common OCR artifacts
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]", "", text)

        return text.strip()


class OlmocrEngine(BaseOCREngine):
    """OlmOCR engine implementation (direct model inference)"""

    def __init__(self, config: OCRConfig, model_config: ModelConfig):
        super().__init__(config, model_config)
        self.model_path = model_config.model_path or "allenai/olmOCR-7B-0225-preview"
        self.processor = None
        self.model = None
        self.device = model_config.device if hasattr(model_config, "device") else "cpu"
        # Initialize preprocessor and postprocessor
        self.preprocessor = OlmocrPreprocessor(config)
        self.postprocessor = OlmocrPostprocessor(config)

    def initialize(self) -> bool:
        try:
            self.logger.info("Initializing OlmOCR engine (direct model inference)")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            torch_dtype = (
                torch.float16
                if torch.cuda.is_available() and self.device == "cuda"
                else torch.float32
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
            ).eval()

            if torch.cuda.is_available() and self.device == "cuda":
                self.model = self.model.to("cuda")
                self.device = "cuda"
                self.logger.info("Using CUDA device")
            else:
                self.device = "cpu"
                self.logger.info("Using CPU device")
            self._initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize OlmOCR: {e}")
            return False

    def _process_olmocr_image(self, image: Image.Image, prompt: str) -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[chat_prompt], images=[image], padding=True, return_tensors="pt"
            )

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Allow configurable generation parameters
            gen_params = {
                "temperature": 0.8,
                "max_new_tokens": 512,
                "num_return_sequences": 1,
                "do_sample": True,
            }
            if (
                hasattr(self.model_config, "custom_params")
                and self.model_config.custom_params
            ):
                gen_params.update(self.model_config.custom_params)
            with torch.no_grad():
                output = self.model.generate(**inputs, **gen_params)
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            text_output = self.processor.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            return text_output[0] if text_output else ""
        except Exception as e:
            self.logger.error(f"Error in OlmOCR model inference: {e}")
            return ""

    def process_image(self, image_path: Union[str, Path]) -> OCRResult:
        self.logger.info(f"Processing image: {image_path}")
        start_time = time.time()
        try:
            # If input is a PDF, process as PDF
            if str(image_path).lower().endswith(".pdf"):
                results = self.process_pdf(image_path)
                return results[0] if results else OCRResult("", [], {}, 0.0, "OlmOCR")

            # Otherwise, treat as image
            image = Image.open(image_path)

            # Resize so longest dimension is 1024
            max_dim = max(image.size)
            if max_dim > 1024:
                ratio = 1024 / max_dim
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Use a generic prompt for images
            prompt = "Extract all text and tables from this document image."
            text = self._process_olmocr_image(image, prompt)
            elements = [
                TextElement(
                    text=text, confidence=0.9, element_type="document", page_number=0
                )
            ]

            processing_time = time.time() - start_time

            result = OCRResult(
                text=text,
                elements=elements,
                metadata={"model_name": "olmOCR-7B", "source_file": str(image_path)},
                processing_time=processing_time,
                model_name="OlmOCR",
            )

            result = self.postprocessor.postprocess(result)
            return result

        except Exception as e:
            self.logger.error(f"Error processing image with OlmOCR: {e}")
            raise

    def process_pdf(self, pdf_path: Union[str, Path]) -> List[OCRResult]:
        start_time = time.time()
        results = []
        try:
            import fitz  # PyMuPDF

            pdf_document = fitz.open(pdf_path)
            for page_num in range(len(pdf_document)):
                # Render page to base64 PNG
                image_base64 = render_pdf_to_base64png(
                    str(pdf_path), page_num + 1, target_longest_image_dim=1024
                )
                # Get anchor text for the page
                anchor_text = get_anchor_text(
                    str(pdf_path),
                    page_num + 1,
                    pdf_engine="pdfreport",
                    target_length=4000,
                )
                prompt = build_finetuning_prompt(anchor_text)
                # Decode base64 image
                main_image = Image.open(BytesIO(base64.b64decode(image_base64)))
                text = self._process_olmocr_image(main_image, prompt)
                elements = [
                    TextElement(
                        text=text,
                        confidence=0.9,
                        element_type="document",
                        page_number=page_num,
                    )
                ]
                page_processing_time = time.time() - start_time
                result = OCRResult(
                    text=text,
                    elements=elements,
                    metadata={
                        "model_name": "olmOCR-7B",
                        "source_file": str(pdf_path),
                        "page": page_num,
                    },
                    processing_time=page_processing_time,
                    model_name="OlmOCR",
                )
                result = self.postprocessor.postprocess(result)
                results.append(result)
            pdf_document.close()
            return results
        except Exception as e:
            self.logger.error(f"Error processing PDF with OlmOCR: {e}")
            raise

    def process_batch(self, file_paths: List[Union[str, Path]]) -> List[OCRResult]:
        results = []
        for file_path in file_paths:
            if str(file_path).lower().endswith(".pdf"):
                results.extend(self.process_pdf(file_path))
            else:
                results.append(self.process_image(file_path))
        return results

    def get_supported_formats(self) -> List[str]:
        return [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "processor") and self.processor is not None:
            del self.processor

        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
