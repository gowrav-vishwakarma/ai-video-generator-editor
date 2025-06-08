# i2v_modules/i2v_ltx.py

import os
import torch
from dataclasses import dataclass
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
# --- We need to import ContentConfig ---
from config_manager import DEVICE, clear_vram_globally, ContentConfig
from PIL import Image
import numpy as np

@dataclass
class I2VConfig:
    """Configuration for the LTX Video Generator pipeline."""
    model_id: str = "Lightricks/LTX-Video"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    width: int = 704
    height: int = 480
    fps: int = 24

I2V_PIPE = None

def get_model_capabilities() -> dict:
    """Returns the optimal settings for the LTX model."""
    return {
        "resolutions": { 
            "Portrait": (480, 704),
            "Landscape": (704, 480)
        },
        "max_chunk_duration": 2.5 
    }

def load_pipeline(config: I2VConfig):
    global I2V_PIPE
    if I2V_PIPE is None:
        print(f"Loading I2V pipeline (LTX): {config.model_id}...")
        I2V_PIPE = LTXImageToVideoPipeline.from_pretrained(config.model_id, torch_dtype=torch.bfloat16)
        I2V_PIPE.enable_model_cpu_offload()
        print("I2V (LTX) pipeline loaded.")
    return I2V_PIPE

def clear_i2v_vram():
    global I2V_PIPE
    print("Clearing I2V (LTX) VRAM...")
    if I2V_PIPE is not None:
        clear_vram_globally(I2V_PIPE)
    I2V_PIPE = None
    print("I2V (LTX) VRAM cleared.")

def _resize_and_pad(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    original_aspect = image.width / image.height
    target_aspect = target_width / target_height
    if original_aspect > target_aspect:
        new_width = target_width; new_height = int(target_width / original_aspect)
    else:
        new_height = target_height; new_width = int(target_height * original_aspect)
    if new_width <= 0 or new_height <= 0: new_width, new_height = 1, 1
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    background = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    background.paste(resized_image, (paste_x, paste_y))
    return background

# #############################################################################
# # --- UPDATED FUNCTION SIGNATURE ---
# #############################################################################
def generate_video_from_image(
    image_path: str,
    output_video_path: str,
    target_duration: float,
    content_config: ContentConfig, # <-- It now accepts the main config
    i2v_config: I2VConfig,
    visual_prompt: str,
    motion_prompt: str = None
) -> str:
    pipe = load_pipeline(i2v_config)
    print(f"I2V (LTX): Received request for chunk with target duration: {target_duration:.2f}s.")

    input_image = load_image(image_path)
    
    target_res_map = get_model_capabilities()["resolutions"]
    aspect_ratio = "Landscape" if input_image.width > input_image.height else "Portrait"
    target_width, target_height = target_res_map[aspect_ratio]
    
    print(f"Preparing input image for LTX target size: {target_width}x{target_height}")
    prepared_image = _resize_and_pad(input_image, target_width, target_height)

    # --- THIS MODULE NOW GETS FPS FROM THE CONFIG ITSELF ---
    fps = content_config.fps
    num_frames = max(16, int(target_duration * fps))
    print(f"Requesting LTX to generate {num_frames} frames at {fps} FPS.")

    full_prompt = f"{visual_prompt}, {motion_prompt}" if motion_prompt else visual_prompt
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    video = pipe(
        prompt=full_prompt,
        image=prepared_image,
        width=target_width,
        height=target_height,
        num_frames=num_frames,
        num_inference_steps=i2v_config.num_inference_steps,
        guidance_scale=i2v_config.guidance_scale,
        negative_prompt=negative_prompt
    ).frames[0]
    
    export_to_video(video, output_video_path, fps=fps)
    
    print(f"LTX video chunk ({len(video)}f) saved to {output_video_path}")
    return output_video_path