# i2v_modules/i2v_svd.py

import os
import torch
from dataclasses import dataclass
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
# --- We need to import ContentConfig to use it in the function signature ---
from config_manager import DEVICE, clear_vram_globally, ContentConfig
from PIL import Image

@dataclass
class I2VConfig:
    """Configuration for Stable Video Diffusion (SVD) model."""
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    decode_chunk_size: int = 8
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    model_native_frames: int = 25
    svd_min_frames: int = 8

I2V_PIPE = None

# #############################################################################
# # --- NEW: Added model capabilities function for consistency ---
# #############################################################################
def get_model_capabilities() -> dict:
    """Returns the optimal settings for the SVD-XT model."""
    return {
        "resolutions": {
            # This model works best with these resolutions.
            "Portrait": (576, 1024),
            "Landscape": (1024, 576),
        },
        # SVD generates 25 frames, which works well for up to 3s clips.
        "max_chunk_duration": 3.0 
    }

def load_pipeline(config: I2VConfig):
    global I2V_PIPE
    if I2V_PIPE is None:
        print(f"Loading I2V pipeline (SVD): {config.model_id}...")
        I2V_PIPE = StableVideoDiffusionPipeline.from_pretrained(
            config.model_id, 
            torch_dtype=torch.float16
            # REMOVED: variant="fp16" for better compatibility
        )
        I2V_PIPE.enable_model_cpu_offload()
        print("I2V (SVD) pipeline loaded.")
    return I2V_PIPE

def clear_i2v_vram():
    global I2V_PIPE
    print("Clearing I2V (SVD) VRAM...")
    if I2V_PIPE is not None:
        clear_vram_globally(I2V_PIPE)
    I2V_PIPE = None
    print("I2V (SVD) VRAM cleared.")

def _resize_and_pad(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Resizes an image to fit within the target dimensions while maintaining aspect ratio,
    then pads the remaining space with black bars.
    """
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
# # --- UPDATED: Function signature now matches the new standard ---
# #############################################################################
def generate_video_from_image(
    image_path: str,
    output_video_path: str,
    target_duration: float,
    content_config: ContentConfig, # <-- Accepts the main config for consistency
    i2v_config: I2VConfig,
    visual_prompt: str, # <-- Kept for signature consistency, though SVD doesn't use it
    motion_prompt: str = None
) -> str:
    pipe = load_pipeline(i2v_config)
    print(f"I2V (SVD): Received request for chunk with target duration: {target_duration:.2f}s.")

    input_image = load_image(image_path)
    
    # Determine optimal SVD size based on input image orientation
    if input_image.width > input_image.height:
        svd_target_width = 1024; svd_target_height = 576
    else:
        svd_target_width = 576; svd_target_height = 1024
        
    print(f"Preparing input image for SVD target size: {svd_target_width}x{svd_target_height}")
    prepared_image = _resize_and_pad(input_image, svd_target_width, svd_target_height)

    frames_to_generate_by_model = i2v_config.model_native_frames
    
    # SVD's core logic: calculate the FPS needed to stretch 25 frames to the target duration
    if target_duration > 0:
        calculated_fps = max(1, round(frames_to_generate_by_model / target_duration))
    else:
        calculated_fps = 8 

    print(f"  SVD will produce {frames_to_generate_by_model} frames.")
    print(f"  Exporting at a calculated {calculated_fps} FPS to meet {target_duration:.2f}s target.")
    if motion_prompt:
        print(f"  Using motion prompt: {motion_prompt}")

    motion_bucket_id = i2v_config.motion_bucket_id
    if motion_prompt:
        motion_prompt_lower = motion_prompt.lower()
        if any(word in motion_prompt_lower for word in ['fast', 'quick', 'rapid', 'dynamic']):
            motion_bucket_id = min(255, motion_bucket_id + 50)
        elif any(word in motion_prompt_lower for word in ['slow', 'gentle', 'subtle', 'smooth']):
            motion_bucket_id = max(0, motion_bucket_id - 50)
        print(f"  Adjusted motion_bucket_id to {motion_bucket_id}")

    video_frames_list = pipe(
        image=prepared_image,
        height=svd_target_height,
        width=svd_target_width,
        decode_chunk_size=i2v_config.decode_chunk_size,
        num_frames=frames_to_generate_by_model,
        motion_bucket_id=motion_bucket_id,
        fps=7,
        noise_aug_strength=i2v_config.noise_aug_strength,
    ).frames[0]

    export_to_video(video_frames_list, output_video_path, fps=calculated_fps)
    
    print(f"SVD video chunk ({len(video_frames_list)}f exported @ {calculated_fps}fps) saved to {output_video_path}")
    return output_video_path