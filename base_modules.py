# In base_modules.py

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field

# --- NEW: Define the ModuleCapabilities Contract ---
class ModuleCapabilities(BaseModel):
    """A standardized spec sheet for all generation modules."""
    
    title: str = Field(description="Title to show in dropdowns")

    # Resource Requirements
    vram_gb_min: float = Field(default=4.0, description="Minimum GPU VRAM required in GB.")
    ram_gb_min: float = Field(default=8.0, description="Minimum system RAM required in GB.")

    # Format & Control Support
    supported_formats: List[Literal["Portrait", "Landscape"]] = Field(default=["Portrait", "Landscape"])
    supports_ip_adapter: bool = Field(default=False, description="True if the module can use IP-Adapter for subject consistency.")
    supports_lora: bool = Field(default=False, description="True if the module supports LoRA weights.")
    
    # Subject & Prompting
    max_subjects: int = Field(default=0, description="Maximum number of distinct subjects/characters the module can handle at once (e.g., via IP-Adapter).")
    accepts_text_prompt: bool = Field(default=True, description="True if the module uses a text prompt.")
    accepts_negative_prompt: bool = Field(default=True, description="True if the module uses a negative prompt.")
    
    # Type-Specific
    supported_tts_languages: List[str] = Field(default=[], description="List of languages supported by a TTS module (e.g., ['en', 'es']).")

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

    # --- NEW: Enforce capabilities contract ---
    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        """Returns the spec sheet for this module."""
        raise NotImplementedError

    @abstractmethod
    def generate_script(self, topic: str, content_config: ContentConfig) -> Dict[str, Any]:
        """Generates the main script, visual prompts, hashtags, and context descriptions."""
        pass

    @abstractmethod
    def generate_chunk_visual_prompts(self, scene_narration: str, original_scene_prompt: str, num_chunks: int, content_config: ContentConfig, main_subject: str, setting: str) -> List[Tuple[str, str]]:
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

    # --- NEW: Enforce capabilities contract ---
    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        """Returns the spec sheet for this module."""
        raise NotImplementedError

    @abstractmethod
    def generate_audio(self, text: str, output_dir: str, scene_idx: int, language: str, speaker_wav: Optional[str] = None) -> Tuple[str, float]:
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
        
    # --- NEW: Enforce capabilities contract ---
    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        """Returns the spec sheet for this module."""
        raise NotImplementedError
        
    @abstractmethod
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Returns a dictionary of the model's capabilities, like resolutions."""
        pass

    def enhance_prompt(self, prompt: str, prompt_type: str = "visual") -> str:
        return prompt

    @abstractmethod
    def clear_vram(self):
        """Clears the VRAM used by the pipeline."""
        pass

class BaseT2I(BaseVideoGen):
    """Abstract Base Class for Text-to-Image modules."""
    @abstractmethod
    def generate_image(self, prompt: str, negative_prompt: str, output_path: str, width: int, height: int, ip_adapter_image: Optional[Union[str, List[str]]] = None, seed: int = -1) -> str:
        """Generates an image from a text prompt, optionally using an IP-Adapter image."""
        pass

class BaseI2V(BaseVideoGen):
    """Abstract Base Class for Image-to-Video modules."""
    @abstractmethod
    def generate_video_from_image(self, image_path: str, output_video_path: str, target_duration: float, content_config: ContentConfig, visual_prompt: str, motion_prompt: Optional[str], ip_adapter_image: Optional[Union[str, List[str]]] = None) -> str:
        """Generates a video from an initial image, optionally using an IP-Adapter image for style/subject."""
        pass

class BaseT2V(BaseVideoGen):
    """Abstract Base Class for Text-to-Video modules."""
    @abstractmethod
    def generate_video_from_text(self, prompt: str, output_video_path: str, num_frames: int, fps: int, width: int, height: int, ip_adapter_image: Optional[Union[str, List[str]]] = None) -> str:
        """Generates a video directly from a text prompt, optionally using an IP-Adapter image."""
        pass