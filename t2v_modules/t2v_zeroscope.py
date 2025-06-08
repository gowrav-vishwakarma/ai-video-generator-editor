# In t2v_modules/t2v_zeroscope.py

import os
import torch
from dataclasses import dataclass
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from config_manager import DEVICE, clear_vram_globally
# We don't need numpy imported here anymore since the output is already a numpy array
# import numpy as np 

@dataclass
class T2VConfig:
    model_id: str = "cerspense/zeroscope_v2_576w"
    num_inference_steps: int = 40
    num_frames: int = 24

T2V_PIPE = None

def get_model_capabilities() -> dict:
    """Returns the optimal settings for the Zeroscope model."""
    return {
        "resolutions": {
            # This model works best at a fixed landscape size
            "Portrait": (576, 320),
            "Landscape": (576, 320),
        },
        # Zeroscope works best with shorter clips
        "max_chunk_duration": 2.0 
    }


def load_pipeline(config: T2VConfig):
    global T2V_PIPE
    if T2V_PIPE is None:
        print(f"Loading T2V pipeline ({config.model_id})...")
        T2V_PIPE = DiffusionPipeline.from_pretrained(config.model_id, torch_dtype=torch.float16)
        T2V_PIPE.scheduler = DPMSolverMultistepScheduler.from_config(T2V_PIPE.scheduler.config)
        T2V_PIPE.enable_model_cpu_offload()
        print(f"T2V ({config.model_id}) pipeline loaded and configured with DPMSolver.")
    return T2V_PIPE

def clear_t2v_vram():
    global T2V_PIPE
    print(f"Clearing T2V VRAM...")
    if T2V_PIPE is not None: clear_vram_globally(T2V_PIPE)
    T2V_PIPE = None
    print("T2V VRAM cleared.")

def generate_video_from_text(
    prompt: str,
    output_video_path: str,
    num_frames: int,
    fps: int,
    width: int,
    height: int,
    t2v_config: T2VConfig
) -> str:
    pipe = load_pipeline(t2v_config)
    print(f"Generating T2V ({width}x{height}) for prompt: \"{prompt[:50]}...\" ({num_frames} frames)")
    
    # This returns a NumPy array of shape (1, F, H, W, C)
    video_frames_batch = pipe(
        prompt=prompt,
        num_inference_steps=t2v_config.num_inference_steps,
        height=height,
        width=width,
        num_frames=num_frames,
        output_type="np"  # Explicitly request NumPy output
    ).frames
    
    # #############################################################################
    # # --- THE DEFINITIVE FIX IS HERE ---
    # # We select the first (and only) video from the batch dimension.
    # # This gives us an object of shape (F, H, W, C), which can be iterated
    # # into a list of frames that export_to_video expects.
    # #############################################################################
    video_frames = video_frames_batch[0]
    
    export_to_video(video_frames, output_video_path, fps=fps)
    # #############################################################################
    
    print(f"T2V video chunk saved to {output_video_path}")
    return output_video_path