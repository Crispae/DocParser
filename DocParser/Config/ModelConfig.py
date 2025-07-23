from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model-specific configuration"""

    model_path: Optional[str] = None
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    device: str = "cpu"
    batch_size: int = 1
    custom_params: Dict[str, Any] = None
    gpu_id: int = 0  # <--- Add this line
