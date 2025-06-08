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
    final_output_resolution: Tuple[int, int] = (1080, 1920)  # Vertical 9:16
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

    # #############################################################################
    # # --- THE MAIN FIX ---
    # # We now set a high-quality default resolution directly.
    # # All complex properties and setters are removed.
    # # This is now the single source of truth for generation resolution.
    # #############################################################################
    generation_resolution: Tuple[int, int] = (1024, 1024)

    def __post_init__(self):
        """Perform validation after the object is created."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Validate that all resolution dimensions are divisible by 8
        for res_type, res_val in [("Generation", self.generation_resolution), ("Final Output", self.final_output_resolution)]:
            if res_val[0] % 8 != 0 or res_val[1] % 8 != 0:
                raise ValueError(f"{res_type} resolution {res_val} must have dimensions divisible by 8.")
        
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