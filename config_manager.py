import os
import torch
import gc
from dataclasses import dataclass, field
from typing import Tuple, Optional

# Global device setting
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@dataclass
class ContentConfig:
    """
    Configuration for overall content generation parameters.
    This is a simple data container. All logic is removed for clarity and direct control.
    """
    # --- Video settings ---
    target_video_length_hint: float = 20.0
    model_max_video_chunk_duration: float = 3.0
    fps: int = 24

    # --- Scene settings (for LLM guidance) ---
    min_scenes: int = 2
    max_scenes: int = 5
    max_scene_narration_duration_hint: float = 6.0

    # --- Model pipeline selection ---
    use_svd_flow: bool = True  # True for T2I -> I2V (SDXL -> SVD)

    # --- Output settings ---
    output_dir: str = "modular_reels_output"
    font_for_subtitles: str = "Arial"

    aspect_ratio_format: str = "Portrait (9:16)" # The user's choice

    @property
    def final_output_resolution(self) -> Tuple[int, int]:
        """Calculates final resolution based on the chosen aspect ratio format."""
        if self.aspect_ratio_format == "Landscape (16:9)":
            return (1920, 1080)
        else: # Default to Portrait
            return (1080, 1920)

    @property
    def generation_resolution(self) -> Tuple[int, int]:
        """Calculates generation resolution for SDXL based on aspect ratio."""
        if self.aspect_ratio_format == "Landscape (16:9)":
            # A common landscape resolution for SDXL
            return (1344, 768) 
        else: # Default to Portrait
            # A common portrait resolution for SDXL
            return (768, 1344)


    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        # We can remove the old validation as the properties handle it.
        print(f"Project Format: {self.aspect_ratio_format}")
        print(f"Using Generation Resolution: {self.generation_resolution[0]}x{self.generation_resolution[1]}")
        print(f"Using Final Output Resolution: {self.final_output_resolution[0]}x{self.final_output_resolution[1]}")


@dataclass
class ModuleSelectorConfig:
    """Selects which implementation module to use for each step."""
    llm_module: str = "llm_modules.llm_zephyr"
    tts_module: str = "tts_modules.tts_coqui"
    t2i_module: str = "t2i_modules.t2i_sdxl"
    i2v_module: str = "i2v_modules.i2v_svd"
    t2v_module: str = "t2v_modules.t2v_zeroscope"


def clear_vram_globally(*models_or_pipelines_to_del):
    """Aggressively tries to clear VRAM."""
    print(f"Attempting to clear VRAM. Received {len(models_or_pipelines_to_del)} items to delete.")
    
    for item_idx, item in enumerate(models_or_pipelines_to_del):
        if item is None: continue
        item_name = type(item).__name__
        if hasattr(item, 'cpu') and callable(item.cpu):
            try:
                item.cpu()
                print(f"Moved {item_name} (item {item_idx}) to CPU.")
            except Exception as e:
                print(f"Warning: Could not move {item_name} (item {item_idx}) to CPU: {e}")
        
    del models_or_pipelines_to_del 
    collected_count = gc.collect()
    print(f"Garbage collector collected {collected_count} objects.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("PyTorch CUDA cache emptied.")
    
    print("VRAM clearing attempt finished.")