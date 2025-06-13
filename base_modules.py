# In base_modules.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field

class ModuleCapabilities(BaseModel):
    title: str = Field(description="Title to show in dropdowns")
    vram_gb_min: float = Field(default=4.0)
    ram_gb_min: float = Field(default=8.0)
    supported_formats: List[Literal["Portrait", "Landscape"]] = Field(default=["Portrait", "Landscape"])
    supports_ip_adapter: bool = Field(default=False)
    supports_lora: bool = Field(default=False)
    max_subjects: int = Field(default=0)
    accepts_text_prompt: bool = Field(default=True)
    accepts_negative_prompt: bool = Field(default=True)
    supported_tts_languages: List[str] = Field(default=[])

class ContentConfig(BaseModel): pass
class ProjectState(BaseModel): pass

class BaseModuleConfig(BaseModel): model_id: str

class BaseLLM(ABC):
    def __init__(self, config: BaseModuleConfig): self.config = config; self.model = None; self.tokenizer = None
    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> ModuleCapabilities: raise NotImplementedError
    @abstractmethod
    def generate_script(self, topic: str, content_config: ContentConfig) -> Dict[str, Any]: pass
    @abstractmethod
    def generate_shot_visual_prompts(self, scene_narration: str, original_scene_prompt: str, num_shots: int, content_config: ContentConfig, main_subject: str, setting: str) -> List[Tuple[str, str]]: pass
    @abstractmethod
    def clear_vram(self): pass

class BaseTTS(ABC):
    def __init__(self, config: BaseModuleConfig): self.config = config; self.model = None
    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> ModuleCapabilities: raise NotImplementedError
    @abstractmethod
    def generate_audio(self, text: str, output_dir: str, scene_idx: int, language: str, speaker_wav: Optional[str] = None) -> Tuple[str, float]: pass
    @abstractmethod
    def clear_vram(self): pass

class BaseVideoGen(ABC):
    def __init__(self, config: BaseModuleConfig): self.config = config; self.pipe = None
    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> ModuleCapabilities: raise NotImplementedError
    
    # --- MODIFIED: Changed to @classmethod ---
    @classmethod
    @abstractmethod
    def get_model_capabilities(cls) -> Dict[str, Any]:
        """Returns a dictionary of the model's capabilities, like resolutions and max duration."""
        raise NotImplementedError
    # --- END OF MODIFICATION ---

    def enhance_prompt(self, prompt: str, prompt_type: str = "visual") -> str: return prompt
    @abstractmethod
    def clear_vram(self): pass

class BaseT2I(BaseVideoGen):
    @abstractmethod
    def generate_image(self, prompt: str, negative_prompt: str, output_path: str, width: int, height: int, ip_adapter_image: Optional[Union[str, List[str]]] = None, seed: int = -1) -> str: pass

class BaseI2V(BaseVideoGen):
    @abstractmethod
    def generate_video_from_image(self, image_path: str, output_video_path: str, target_duration: float, content_config: ContentConfig, visual_prompt: str, motion_prompt: Optional[str], ip_adapter_image: Optional[Union[str, List[str]]] = None) -> str: pass

class BaseT2V(BaseVideoGen):
    @abstractmethod
    def generate_video_from_text(self, prompt: str, output_video_path: str, num_frames: int, fps: int, width: int, height: int, ip_adapter_image: Optional[Union[str, List[str]]] = None) -> str: pass