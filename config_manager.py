# In config_manager.py
import os
import torch
import gc
from pydantic import BaseModel, Field
from typing import Dict, Tuple, Literal, List, Optional
from uuid import UUID, uuid4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ContentConfig(BaseModel):
    fps: int = 24
    output_dir: str = "modular_reels_output"
    font_for_subtitles: str = "Arial"
    add_narration_text_to_video: bool = True
    aspect_ratio_format: Literal["Portrait", "Landscape"] = "Landscape"
    
    @property
    def final_output_resolution(self) -> Tuple[int, int]:
        return (1080, 1920) if self.aspect_ratio_format == "Portrait" else (1920, 1080)

class CharacterVersion(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    name: str = "base"
    reference_image_path: str
    t2i_module_path: Optional[str] = None
    t2i_prompt: Optional[str] = None
    status: str = "completed"

class Character(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    name: str
    versions: List[CharacterVersion] = Field(default_factory=list)

class Voice(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    name: str
    tts_module_path: str
    reference_wav_path: str
    
class Narration(BaseModel):
    text: str = ""
    voice_uuid: Optional[UUID] = None
    audio_path: Optional[str] = None
    duration: float = 0.0
    status: str = "pending"

class Shot(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    # --- START OF MODIFICATION ---
    generation_flow: Literal["T2I_I2V", "T2V", "Upload_I2V"] = "T2I_I2V"
    uploaded_image_path: Optional[str] = None
    # --- END OF MODIFICATION ---
    
    visual_prompt: str = "A cinematic shot"
    motion_prompt: str = "Subtle camera movement"
    module_selections: Dict[str, str] = Field(default_factory=dict)
    character_uuids: List[UUID] = Field(default_factory=list)
    keyframe_image_path: Optional[str] = None
    video_path: Optional[str] = None
    status: str = "pending"

class Scene(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    # --- START OF MODIFICATION ---
    title: str = "Untitled Scene"
    # --- END OF MODIFICATION ---
    narration: Narration = Field(default_factory=Narration)
    shots: List[Shot] = Field(default_factory=list)
    assembled_video_path: Optional[str] = None
    status: str = "pending"

class ProjectState(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    title: str
    video_format: Literal["Portrait", "Landscape"] = "Landscape"
    characters: List[Character] = Field(default_factory=list)
    voices: List[Voice] = Field(default_factory=list)
    scenes: List[Scene] = Field(default_factory=list)
    final_video_path: Optional[str] = None
    status: str = "in_progress"
    add_narration_text_to_video: bool = True
    
    @property
    def final_output_resolution(self) -> Tuple[int, int]:
        return (1080, 1920) if self.video_format == "Portrait" else (1920, 1080)

def clear_vram_globally(*items_to_del):
    print(f"Attempting to clear VRAM. Received {len(items_to_del)} items to delete.")
    for item in items_to_del:
        if hasattr(item, 'to'):
            try: item.to('cpu')
            except Exception: pass
    del items_to_del
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()