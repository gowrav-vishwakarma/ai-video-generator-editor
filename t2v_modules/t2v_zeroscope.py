# t2v_modules/t2v_zeroscope.py
import os
import torch
from dataclasses import dataclass
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from config_manager import DEVICE, clear_vram_globally

@dataclass
class T2VConfig:
    """Configuration for Zeroscope text-to-video model."""
    model_id: str = "cerspense/zeroscope_v2_576w"
    num_inference_steps: int = 25
    # num_frames for T2V models is often more flexible than SVD

T2V_PIPE = None

def load_pipeline(config: T2VConfig):
    global T2V_PIPE
    if T2V_PIPE is None:
        print(f"Loading T2V pipeline ({config.model_id})...")
        T2V_PIPE = DiffusionPipeline.from_pretrained(
            config.model_id, torch_dtype=torch.float16
        )
        T2V_PIPE.enable_model_cpu_offload() # Good for T2V models too
        print(f"T2V ({config.model_id}) pipeline loaded.")
    return T2V_PIPE

def clear_t2v_vram():
    global T2V_PIPE
    print(f"Clearing T2V VRAM...")
    models_to_clear = []
    if T2V_PIPE is not None:
        models_to_clear.append(T2V_PIPE)
    
    clear_vram_globally(*models_to_clear)
    T2V_PIPE = None
    print("T2V VRAM cleared.")

def generate_video_from_text(
    prompt: str,
    output_video_path: str,
    num_frames: int, # Calculated desired frames
    fps: int,
    width: int, # Target width for T2V model
    height: int, # Target height for T2V model
    t2v_config: T2VConfig
) -> str:
    pipe = load_pipeline(t2v_config) # Ensures pipeline is loaded
    print(f"Generating T2V ({width}x{height}) for prompt: \"{prompt[:50]}...\" ({num_frames} frames)")

    # Some T2V models might have their own constraints on num_frames, adjust if necessary
    # For Zeroscope, num_frames is usually flexible.
    
    video_frames = pipe(
        prompt=prompt,
        num_inference_steps=t2v_config.num_inference_steps,
        num_frames=num_frames,
        height=height, # Model specific, e.g. Zeroscope 576w is 320 height
        width=width    # Model specific, e.g. Zeroscope 576w is 576 width
    ).frames 
    # Note: Some T2V pipelines might return a list of frames, others a tensor.
    # The .frames attribute is common for diffusers pipelines returning video.
    # If it's `frames[0]` like SVD, adjust accordingly. Zeroscope usually returns list of PIL.

    export_to_video(video_frames, output_video_path, fps=fps)
    print(f"T2V video chunk saved to {output_video_path}")
    return output_video_path