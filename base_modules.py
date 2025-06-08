# base_modules.py

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel

# Forward-declare to avoid circular imports
class ContentConfig(BaseModel): pass
class ProjectState(BaseModel): pass

# --- Base Configuration Models ---

class BaseModuleConfig(BaseModel):
    """Base for all module-specific configurations."""
    model_id: str

# --- Base Module Classes ---

class BaseLLM(ABC):
    """Abstract Base Class for Language Model modules."""
    def __init__(self, config: BaseModuleConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def generate_script(self, topic: str, content_config: ContentConfig) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Generates the main script, visual prompts, and hashtags for the video."""
        pass

    @abstractmethod
    def generate_chunk_visual_prompts(self, scene_narration: str, original_scene_prompt: str, num_chunks: int, content_config: ContentConfig) -> List[Tuple[str, str]]:
        """Generates visual and motion prompts for each chunk within a scene."""
        pass

    @abstractmethod
    def clear_vram(self):
        """Clears the VRAM used by the model and tokenizer."""
        pass

class BaseTTS(ABC):
    """Abstract Base Class for Text-to-Speech modules."""
    def __init__(self, config: BaseModuleConfig):
        self.config = config
        self.model = None

    @abstractmethod
    def generate_audio(self, text: str, output_dir: str, scene_idx: int, speaker_wav: Optional[str] = None) -> Tuple[str, float]:
        """Generates audio from text."""
        pass

    @abstractmethod
    def clear_vram(self):
        """Clears the VRAM used by the TTS model."""
        pass

class BaseVideoGen(ABC):
    """A common base for all video generation modules (T2I, I2V, T2V)."""
    def __init__(self, config: BaseModuleConfig):
        self.config = config
        self.pipe = None
        
    @abstractmethod
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Returns a dictionary of the model's capabilities, like resolutions."""
        pass

    @abstractmethod
    def clear_vram(self):
        """Clears the VRAM used by the pipeline."""
        pass

class BaseT2I(BaseVideoGen):
    """Abstract Base Class for Text-to-Image modules."""
    @abstractmethod
    def generate_image(self, prompt: str, output_path: str, width: int, height: int) -> str:
        """Generates an image from a text prompt."""
        pass

class BaseI2V(BaseVideoGen):
    """Abstract Base Class for Image-to-Video modules."""
    @abstractmethod
    def generate_video_from_image(self, image_path: str, output_video_path: str, target_duration: float, content_config: ContentConfig, visual_prompt: str, motion_prompt: Optional[str]) -> str:
        """Generates a video from an initial image."""
        pass

class BaseT2V(BaseVideoGen):
    """Abstract Base Class for Text-to-Video modules."""
    @abstractmethod
    def generate_video_from_text(self, prompt: str, output_video_path: str, num_frames: int, fps: int, width: int, height: int) -> str:
        """Generates a video directly from a text prompt."""
        pass