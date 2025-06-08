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
    """Configuration for overall content generation parameters."""
    # --- Video settings ---
    # These are now populated by user input, with these values serving as fallbacks.
    target_video_length_hint: float = 20.0
    min_scenes: int = 2
    max_scenes: int = 5
    aspect_ratio_format: str = "Portrait (9:16)"

    # --- Static settings ---
    model_max_video_chunk_duration: float = 3.0
    fps: int = 24
    use_svd_flow: bool = True
    output_dir: str = "modular_reels_output"
    font_for_subtitles: str = "Arial"

    @property
    def max_scene_narration_duration_hint(self) -> float:
        """Calculates an ideal narration length per scene based on total length and scene count."""
        # Aim for a duration that fits within the target length
        if self.max_scenes > 0:
            return round(self.target_video_length_hint / ((self.min_scenes + self.max_scenes) / 2), 1)
        return 6.0 # Fallback

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
            return (1344, 768)
        else: # Default to Portrait
            return (896, 1152) # A slightly more standard vertical resolution

    def __post_init__(self):
        """Perform validation after the object is created."""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Project Format: {self.aspect_ratio_format}, Target Length: {self.target_video_length_hint}s")
        print(f"Scene Count Range: {self.min_scenes}-{self.max_scenes}")
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