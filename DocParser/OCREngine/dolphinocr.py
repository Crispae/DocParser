from typing import Union, List, Tuple, Optional
from pathlib import Path
import logging
import time
import torch
from PIL import Image
import io
import re
import cv2
import numpy as np

from DocParser.OCREngine.BaseOCREngine import BaseOCREngine
from DocParser.Config.OCRConfig import OCRConfig
from DocParser.Config.ModelConfig import ModelConfig
from DocParser.Config.DataModels import OCRResult, TextElement, BoundingBox
from DocParser.Preprocessor.BasePreprocessor import BasePreprocessor
from DocParser.Postprocessor.BasePostprocessor import BasePostprocessor

from transformers import AutoProcessor, VisionEncoderDecoderModel


# Helper: parse layout string (copied from utils)
def parse_layout_string(layout_str):
    import re

    pattern = re.compile(r"\[(.*?)\]\s*(\w+)")
    results = []
    for match in pattern.finditer(layout_str):
        bbox_str, label = match.groups()
        try:
            # Remove any non-numeric, non-comma, non-dot, non-minus characters
            bbox_clean = re.sub(r"[^0-9.,-]", "", bbox_str)
            bbox = [float(x) for x in bbox_clean.split(",") if x]
            if len(bbox) == 4:
                results.append((bbox, label))
        except Exception:
            continue  # Skip invalid bboxes
    return results


# Helper: crop region from PIL image using normalized bbox
def crop_region(image, bbox):
    w, h = image.size
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)
    return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


class DolphinPreprocessor(BasePreprocessor):
    """Preprocessor for Dolphin OCR engine"""

    def __init__(self, config: OCRConfig):
        ##super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Preprocess image for Dolphin OCR"""
        try:
            # Convert to RGB if necessary
            # if image.mode != "RGB":
            #    image = image.convert("RGB")

            # Dolphin works best with high-resolution images
            # Resize if image is too small for better recognition
            max_size = 896
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            self.logger.error(f"Error in Dolphin preprocessing: {e}")
            return image


class DolphinPostprocessor(BasePostprocessor):
    """Postprocessor for Dolphin OCR engine"""

    def __init__(self, config: OCRConfig):
        ##super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def postprocess(self, result: OCRResult) -> OCRResult:
        """Postprocess Dolphin OCR results"""
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
            self.logger.error(f"Error in Dolphin postprocessing: {e}")
            return result

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common OCR artifacts
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]", "", text)

        return text.strip()


class DolphinEngine(BaseOCREngine):
    """Dolphin OCR engine implementation using Hugging Face model (two-stage pipeline)"""

    def __init__(self, config: OCRConfig, model_config: ModelConfig):
        super().__init__(config, model_config)
        self.model = None
        self.processor = None
        self.device = model_config.device if hasattr(model_config, "device") else "cpu"
        self.preprocessor = DolphinPreprocessor(config)
        self.postprocessor = DolphinPostprocessor(config)

    def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Dolphin OCR engine with Hugging Face model")
            model_name = self.model_config.model_path or "ByteDance/Dolphin"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.eval()
            if torch.cuda.is_available() and self.device == "cuda":
                self.model = self.model.to("cuda")
                self.device = "cuda"
                self.model = self.model.half()
                self.logger.info("Using CUDA device")
            else:
                self.device = "cpu"
                self.logger.info("Using CPU device")
            self._initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Dolphin: {e}")
            return False

    def _model_chat(self, prompt, image):
        # Run the model on a single image and prompt, return cleaned string
        batch_inputs = self.processor(images=image, return_tensors="pt", padding=True)
        batch_pixel_values = batch_inputs.pixel_values.to(self.device)
        if self.device == "cuda":
            batch_pixel_values = batch_pixel_values.half()
        prompts = [f"<s>{prompt} <Answer/>"]
        tokenizer = self.processor.tokenizer
        prompt_inputs = tokenizer(
            prompts, add_special_tokens=False, return_tensors="pt"
        )
        prompt_ids = prompt_inputs.input_ids.to(self.device)
        attention_mask = prompt_inputs.attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values=batch_pixel_values,
                decoder_input_ids=prompt_ids,
                decoder_attention_mask=attention_mask,
                min_length=1,
                max_length=4096,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.1,
            )
        sequences = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
        # Clean prompt and special tokens
        results = []
        for i, sequence in enumerate(sequences):
            cleaned = (
                sequence.replace(prompts[i], "")
                .replace("<pad>", "")
                .replace("</s>", "")
                .strip()
            )
            results.append(cleaned)
        return results[0]  # Only one image at a time in this pipeline

    def process_image(self, image_path: Union[str, Path]) -> OCRResult:
        self.logger.info(f"Processing image: {image_path}")
        start_time = time.time()
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.preprocessor.preprocess(image)
            # Stage 1: Layout analysis
            layout_str = self._model_chat(
                "Parse the reading order of this document.", image
            )
            self.logger.info(f"Layout string: {layout_str}")
            layout_results = parse_layout_string(layout_str)
            self.logger.info(f"Parsed layout regions: {layout_results}")
            elements = []
            full_text = []
            for bbox, label in layout_results:
                self.logger.info(f"Processing region: {label}, bbox: {bbox}")
                crop, abs_bbox = crop_region(image, bbox)
                self.logger.info(f"Cropped region size: {crop.size}")
                if label == "tab":
                    content = self._model_chat("Parse the table in the image.", crop)
                    element_type = "table"
                else:
                    content = self._model_chat("Read text in the image.", crop)
                    element_type = "text"
                self.logger.info(f"Content extracted: {content}")
                elements.append(
                    TextElement(
                        text=content,
                        confidence=0.92,
                        bbox=abs_bbox,
                        element_type=element_type,
                        page_number=0,
                    )
                )
                full_text.append(content)
            processing_time = time.time() - start_time
            result = OCRResult(
                text="\n".join(full_text),
                elements=elements,
                metadata={"model_name": "Dolphin", "source_file": str(image_path)},
                processing_time=processing_time,
                model_name="Dolphin",
            )
            result = self.postprocessor.postprocess(result)
            return result
        except Exception as e:
            self.logger.error(f"Error processing image with Dolphin: {e}")
            raise

    def process_pdf(self, pdf_path: Union[str, Path]) -> List[OCRResult]:
        import fitz  # PyMuPDF

        start_time = time.time()
        results = []
        try:
            pdf_document = fitz.open(pdf_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                image = self.preprocessor.preprocess(image)
                layout_str = self._model_chat(
                    "Parse the reading order of this document.", image
                )
                layout_results = parse_layout_string(layout_str)
                elements = []
                full_text = []
                for bbox, label in layout_results:
                    crop, abs_bbox = crop_region(image, bbox)
                    if label == "tab":
                        content = self._model_chat(
                            "Parse the table in the image.", crop
                        )
                        element_type = "table"
                    else:
                        content = self._model_chat("Read text in the image.", crop)
                        element_type = "text"
                    elements.append(
                        TextElement(
                            text=content,
                            confidence=0.92,
                            bbox=abs_bbox,
                            element_type=element_type,
                            page_number=page_num,
                        )
                    )
                    full_text.append(content)
                page_processing_time = time.time() - start_time
                result = OCRResult(
                    text="\n".join(full_text),
                    elements=elements,
                    metadata={
                        "model_name": "Dolphin",
                        "source_file": str(pdf_path),
                        "page": page_num,
                    },
                    processing_time=page_processing_time,
                    model_name="Dolphin",
                )
                result = self.postprocessor.postprocess(result)
                results.append(result)
            pdf_document.close()
            return results
        except Exception as e:
            self.logger.error(f"Error processing PDF with Dolphin: {e}")
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
