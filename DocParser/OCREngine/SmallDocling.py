from typing import Union, List, Tuple, Optional
from pathlib import Path
import logging
import time
from PIL import Image
import io
import re

from DocParser.OCREngine.BaseOCREngine import BaseOCREngine
from DocParser.Config.OCRConfig import OCRConfig
from DocParser.Config.ModelConfig import ModelConfig
from DocParser.Config.DataModels import OCRResult, TextElement, BoundingBox
from DocParser.Preprocessor.BasePreprocessor import BasePreprocessor
from DocParser.Postprocessor.BasePostprocessor import BasePostprocessor
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch


class SmallDoclingPreprocessor(BasePreprocessor):
    """Preprocessor for SmallDocling OCR engine"""

    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Preprocess image for SmallDocling OCR"""
        try:
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # SmallDocling works well with standard image sizes
            # Resize if image is too large for memory efficiency
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Enhance image quality for better VLM performance
            if self.config.preserve_layout:
                from PIL import ImageEnhance

                # Enhance brightness slightly
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.05)

                # Enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)

            return image

        except Exception as e:
            self.logger.error(f"Error in SmallDocling preprocessing: {e}")
            return image

    def preprocess_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        """Batch preprocess images for SmallDocling OCR"""
        return [self.preprocess(img) for img in images]


class SmallDoclingPostprocessor(BasePostprocessor):
    """Postprocessor for SmallDocling OCR engine"""

    def __init__(self, config: OCRConfig):
        # super().__init__(config)  # Remove this line
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config  # Store config if needed

    def postprocess(self, result: OCRResult) -> OCRResult:
        """Postprocess SmallDocling OCR results"""
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

            # Parse DocTags format if present
            if result.text and self._contains_doctags(result.text):
                result = self._parse_doctags_structure(result)

            return result

        except Exception as e:
            self.logger.error(f"Error in SmallDocling postprocessing: {e}")
            return result

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common OCR artifacts
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]", "", text)

        return text.strip()

    def _contains_doctags(self, text: str) -> bool:
        """Check if text contains DocTags format"""
        return "<" in text and ">" in text

    def _parse_doctags_structure(self, result: OCRResult) -> OCRResult:
        """Parse DocTags format into structured elements"""
        # This is a simplified DocTags parser
        # In reality, DocTags has more complex structure
        lines = result.text.split("\n")
        structured_elements = []

        for i, line in enumerate(lines):
            if line.strip():
                # Basic DocTags parsing
                if "<" in line and ">" in line:
                    # Extract text from tags
                    import re

                    text_content = re.sub(r"<[^>]+>", "", line).strip()
                    if text_content:
                        element = TextElement(
                            text=text_content,
                            confidence=0.95,
                            element_type="structured_text",
                            page_number=0,
                        )
                        structured_elements.append(element)
                else:
                    # Regular text
                    element = TextElement(
                        text=line.strip(),
                        confidence=0.95,
                        element_type="text",
                        page_number=0,
                    )
                    structured_elements.append(element)

        if structured_elements:
            result.elements = structured_elements
            result.text = " ".join([elem.text for elem in structured_elements])

        return result


class SmallDoclingEngine(BaseOCREngine):
    """SmolDocling OCR engine implementation"""

    def __init__(self, config: OCRConfig, model_config: ModelConfig):
        super().__init__(config, model_config)
        print("DEBUG: model_config.device =", model_config.device)

        self.model = None
        self.processor = None
        self.device = model_config.device
        # Use gpu_id if present, else default to 0
        self.gpu_id = (
            getattr(model_config, "gpu_id", 0) if self.device == "cuda" else None
        )
        # Initialize preprocessor and postprocessor
        self.preprocessor = SmallDoclingPreprocessor(config)
        self.postprocessor = SmallDoclingPostprocessor(config)

    def initialize(self) -> bool:
        try:
            self.logger.info("Initializing SmolDocling engine")

            model_name = (
                self.model_config.model_path or "ds4sd/SmolDocling-256M-preview"
            )

            self.processor = AutoProcessor.from_pretrained(model_name)

            if self.device == "cuda":
                device = torch.device(f"cuda:{self.gpu_id}")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map={"": self.gpu_id},
                )
                self.model = self.model.to(device)
                self._torch_device = device
                print("Model device:", next(self.model.parameters()).device)
            else:
                print("Model is loading in CPU")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                )
                self._torch_device = torch.device("cpu")

            self._initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize SmolDocling: {e}")
            return False

    def process_image(self, image_path: Union[str, Path]) -> OCRResult:
        start_time = time.time()
        try:
            import torch

            image = Image.open(image_path)
            image = self.preprocessor.preprocess(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Convert this page to docling."},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            # Move tensors to the correct device
            inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
            inputs.pop("rows", None)
            inputs.pop("cols", None)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=2048, do_sample=False
                )
            prompt_length = inputs["input_ids"].shape[1]
            trimmed_outputs = outputs[:, prompt_length:]
            generated_text = self.processor.decode(
                trimmed_outputs[0], skip_special_tokens=True
            )
            elements = self._parse_doctags(generated_text)
            processing_time = time.time() - start_time
            result = OCRResult(
                text=generated_text,
                elements=elements,
                metadata={
                    "model_name": "SmolDocling-256M",
                    "image_path": str(image_path),
                },
                processing_time=processing_time,
                model_name="SmolDocling",
                confidence_score=0.95,
            )
            result = self.postprocessor.postprocess(result)
            return result
        except Exception as e:
            self.logger.error(f"Error processing image with SmolDocling: {e}")
            raise

    def process_pdf(self, pdf_path: Union[str, Path]) -> List[OCRResult]:
        import fitz  # PyMuPDF
        import torch

        results = []
        try:
            pdf_document = fitz.open(pdf_path)
            images = []
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
            pdf_document.close()
            batch_size = getattr(self.model_config, "batch_size", 3) or 3
            if not isinstance(batch_size, int) or batch_size < 1:
                batch_size = 3
            images = self.preprocessor.preprocess_batch(images)
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                messages = [
                    {
                        "role": "user",
                        "content": [
                            *([{"type": "image"}] * len(batch_images)),
                            {"type": "text", "text": "Convert these pages to docling."},
                        ],
                    }
                ]
                prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=prompt, images=batch_images, return_tensors="pt"
                )
                # Move tensors to the correct device
                inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
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
                page_texts = [
                    t.strip() for t in re.split(r"<page_break>", generated_text)
                ]
                for j, page_text in enumerate(page_texts):
                    elements = self._parse_doctags(page_text)
                    result = OCRResult(
                        text=page_text,
                        elements=elements,
                        metadata={
                            "model_name": "SmolDocling-256M",
                            "image_path": str(pdf_path),
                            "page_number": i + j,
                        },
                        processing_time=0.0,
                        model_name="SmolDocling",
                        confidence_score=0.95,
                    )
                    result = self.postprocessor.postprocess(result)
                    results.append(result)
            return results
        except Exception as e:
            self.logger.error(f"Error processing PDF with SmolDocling: {e}")
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

    def _parse_doctags(self, doctags_text: str) -> List[TextElement]:
        """Parse DocTags format into TextElement objects"""
        elements = []
        lines = doctags_text.split("\n")

        for i, line in enumerate(lines):
            if line.strip():
                # Basic parsing - in reality, DocTags has more structure
                element = TextElement(
                    text=line.strip(),
                    confidence=0.95,
                    element_type="text",
                    page_number=0,
                )

                elements.append(element)

        return elements

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "processor") and self.processor is not None:
            del self.processor

        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
