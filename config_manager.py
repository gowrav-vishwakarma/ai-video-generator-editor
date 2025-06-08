# In config_manager.py
import os
import torch
import gc
from pydantic import BaseModel, Field
from typing import Tuple, Literal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda": os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class ContentConfig(BaseModel):
    """Configuration for overall content generation parameters, using Pydantic."""
    # --- User-defined settings from the UI ---
    target_video_length_hint: float = 20.0
    min_scenes: int = 2
    max_scenes: int = 5
    aspect_ratio_format: Literal["Portrait", "Landscape"] = "Portrait"
    use_svd_flow: bool = True

    # --- Static project-wide settings ---
    fps: int = 24
    output_dir: str = "modular_reels_output"
    font_for_subtitles: str = "Arial"

    # --- DYNAMIC settings, to be populated by the TaskExecutor ---
    model_max_video_chunk_duration: float = 2.0 # A safe default
    generation_resolution: Tuple[int, int] = (1024, 1024) # A safe default

    @property
    def max_scene_narration_duration_hint(self) -> float:
        """Calculates an ideal narration length per scene based on total length and scene count."""
        if self.max_scenes > 0 and self.min_scenes > 0:
            avg_scenes = (self.min_scenes + self.max_scenes) / 2
            return round(self.target_video_length_hint / avg_scenes, 1)
        return 6.0 # A safe fallback if scene counts are zero

    @property
    def final_output_resolution(self) -> Tuple[int, int]:
        """Calculates final resolution based on the simplified aspect ratio format."""
        if self.aspect_ratio_format == "Landscape":
            return (1920, 1080)
        return (1080, 1920) # Default to Portrait

    def __init__(self, **data):
        super().__init__(**data)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"--- Project Config Initialized ---")
        print(f"Flow: {'T2I -> I2V (SVD)' if self.use_svd_flow else 'Direct T2V'}")
        print(f"Format: {self.aspect_ratio_format}")
        print(f"Initial Generation Resolution: {self.generation_resolution}")
        print(f"Initial Max Chunk Duration: {self.model_max_video_chunk_duration}s")
        print(f"---------------------------------")


class ModuleSelectorConfig(BaseModel):
    llm_module: str = "llm_modules.llm_zephyr.ZephyrLLM"
    tts_module: str = "tts_modules.tts_coqui.CoquiTTSModule"
    t2i_module: str = "t2i_modules.t2i_juggernaut.JuggernautT2I"
    i2v_module: str = "i2v_modules.i2v_ltx.LtxI2V"
    t2v_module: str = "t2v_modules.t2v_zeroscope.ZeroscopeT2V"
    # t2v_module: str = "t2v_modules.t2v_ltx.LtxT2V"

def clear_vram_globally(*items_to_del):
    print(f"Attempting to clear VRAM. Received {len(items_to_del)} items to delete.")
    for item in items_to_del:
        if hasattr(item, 'to'):
            try:
                item.to('cpu')
            except Exception as e:
                print(f"Could not move item of type {type(item)} to CPU: {e}")
    del items_to_del
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("VRAM clearing attempt finished.")