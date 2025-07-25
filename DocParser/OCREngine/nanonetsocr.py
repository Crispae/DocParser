from typing import Union, List, Tuple, Optional
from pathlib import Path
import logging
import time
import torch
from PIL import Image
import io

from DocParser.OCREngine.BaseOCREngine import BaseOCREngine
from DocParser.Config.OCRConfig import OCRConfig
from DocParser.Config.ModelConfig import ModelConfig
from DocParser.Config.DataModels import OCRResult, TextElement
from DocParser.Preprocessor.BasePreprocessor import BasePreprocessor
from DocParser.Postprocessor.BasePostprocessor import BasePostprocessor


class NanonetsPreprocessor(BasePreprocessor):
    """Preprocessor for Nanonets OCR engine"""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Preprocess image for Nanonets OCR"""
        try:
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if image is too large (Nanonets has input size limits)
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Enhance contrast if needed
            if self.config.preserve_layout:
                # Apply slight contrast enhancement for better text recognition
                from PIL import ImageEnhance

                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)

            return image

        except Exception as e:
            self.logger.error(f"Error in Nanonets preprocessing: {e}")
            return image


class NanonetsPostprocessor(BasePostprocessor):
    """Postprocessor for Nanonets OCR engine"""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def postprocess(self, result: OCRResult) -> OCRResult:
        """Postprocess Nanonets OCR results"""
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
            self.logger.error(f"Error in Nanonets postprocessing: {e}")
            return result

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        import re

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common OCR artifacts
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]", "", text)

        return text.strip()


class NanonetsEngine(BaseOCREngine):
    """Nanonets OCR engine implementation using Hugging Face model"""

    def __init__(self, config: OCRConfig, model_config: ModelConfig):
        super().__init__(config, model_config)
        self.model_name = "nanonets/Nanonets-OCR-s"
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.device = model_config.device if hasattr(model_config, "device") else "cpu"

        # Initialize preprocessor and postprocessor
        self.preprocessor = NanonetsPreprocessor(config)
        self.postprocessor = NanonetsPostprocessor(config)

    def initialize(self) -> bool:
        try:
            from transformers import (
                AutoTokenizer,
                AutoProcessor,
                AutoModelForImageTextToText,
            )

            self.logger.info("Initializing Nanonets OCR engine with Hugging Face model")

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, use_fast=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                ##attn_implementation="flash_attention_2",
            )

            # Do not move model to device manually if using device_map="auto"
            if torch.cuda.is_available() and self.device == "cuda":
                self.logger.info(
                    "Using CUDA device (Accelerate will manage device placement)"
                )
            else:
                self.device = "cpu"
                self.logger.info(
                    "Using CPU device (Accelerate will manage device placement)"
                )

            self.model.eval()  # Set to evaluation mode

            self._initialized = True
            self.logger.info(f"Successfully loaded Nanonets model: {self.model_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Nanonets: {e}")
            return False

    def _extract_text_from_image(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text from a PIL image using the Nanonets model (chat template approach)"""
        try:
            prompt = (
                "Extract the text from the above document as if you were reading it naturally. "
                "Return the tables in html format. Return the equations in LaTeX representation. "
                "If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. "
                "Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. "
                "Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
                "Prefer using ☐ and ☑ for check boxes."
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            chat_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[chat_prompt], images=[image], padding=True, return_tensors="pt"
            )
            # Move tensors to the correct device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Remove unused keys if present
            inputs.pop("rows", None)
            inputs.pop("cols", None)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=4096, do_sample=False
                )
            prompt_length = inputs["input_ids"].shape[1]
            trimmed_outputs = outputs[:, prompt_length:]
            generated_text = self.processor.decode(
                trimmed_outputs[0], skip_special_tokens=True
            )
            confidence = 0.95  # Default confidence
            return generated_text, confidence
        except Exception as e:
            self.logger.error(f"Error extracting text from image: {e}")
            raise

    def process_image(self, image_path: Union[str, Path]) -> OCRResult:
        start_time = time.time()

        try:
            # Load image
            image = Image.open(image_path)

            # Preprocess image
            image = self.preprocessor.preprocess(image)

            # Extract text
            extracted_text, confidence = self._extract_text_from_image(image)

            # Create TextElement (without bounding box since model doesn't provide it)
            element = TextElement(
                text=extracted_text,
                confidence=confidence,
                bbox=None,  # Nanonets-OCR-s doesn't provide bounding boxes
                element_type="text",
            )

            processing_time = time.time() - start_time

            result = OCRResult(
                text=extracted_text,
                elements=[element] if extracted_text else [],
                metadata={"model_name": "Nanonets-OCR-s", "device": self.device},
                processing_time=processing_time,
                model_name="Nanonets-OCR-s",
            )

            # Postprocess result
            result = self.postprocessor.postprocess(result)

            return result

        except Exception as e:
            self.logger.error(f"Error processing image with Nanonets: {e}")
            raise

    def process_pdf(self, pdf_path: Union[str, Path]) -> List[OCRResult]:
        import fitz  # PyMuPDF

        start_time = time.time()
        results = []

        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)

                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # Scale factor for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))

                # Preprocess image
                image = self.preprocessor.preprocess(image)

                # Extract text
                extracted_text, confidence = self._extract_text_from_image(image)

                # Create TextElement
                element = TextElement(
                    text=extracted_text,
                    confidence=confidence,
                    bbox=None,
                    element_type="text",
                    page_number=page_num,
                )

                page_processing_time = time.time() - start_time

                result = OCRResult(
                    text=extracted_text,
                    elements=[element] if extracted_text else [],
                    metadata={
                        "model_name": "Nanonets-OCR-s",
                        "page": page_num,
                        "device": self.device,
                    },
                    processing_time=page_processing_time,
                    model_name="Nanonets-OCR-s",
                )

                # Postprocess result
                result = self.postprocessor.postprocess(result)
                results.append(result)

            pdf_document.close()
            return results

        except Exception as e:
            self.logger.error(f"Error processing PDF with Nanonets: {e}")
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
        return [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "processor") and self.processor is not None:
            del self.processor

        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
