# i2v_modules/i2v_svd.py
import os
import torch
from dataclasses import dataclass
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from config_manager import DEVICE, clear_vram_globally

@dataclass
class I2VConfig:
    """Configuration for Stable Video Diffusion (SVD) model."""
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt" # SVD-XT
    decode_chunk_size: int = 8 # SVD specific, can be 2, 4, 8. Higher might be faster but more VRAM.
    motion_bucket_id: int = 127 
    noise_aug_strength: float = 0.02
    # SVD models have fixed output frames, e.g., SVD-XT is 25, base SVD is 14.
    # The module will use this as the target and clip/loop if necessary.
    # This 'num_frames' is what the model *generates*. The main script calculates desired frames based on duration.
    model_native_frames: int = 25 # For SVD-XT. For base SVD, it's 14.
    # Min frames for generation request to SVD, even if target_chunk_duration is very short
    min_request_frames: int = 8 # SVD might have issues with too few frames requested.
    svd_min_frames: int = 8 # Minimum frames to request from SVD model

I2V_PIPE = None

def load_pipeline(config: I2VConfig):
    global I2V_PIPE
    if I2V_PIPE is None:
        print(f"Loading I2V pipeline (SVD): {config.model_id}...")
        I2V_PIPE = StableVideoDiffusionPipeline.from_pretrained(
            config.model_id, torch_dtype=torch.float16, variant="fp16"
        )
        # SVD is memory hungry, CPU offload is good
        I2V_PIPE.enable_model_cpu_offload() 
        print("I2V (SVD) pipeline loaded.")
    return I2V_PIPE

def clear_i2v_vram():
    global I2V_PIPE
    print("Clearing I2V (SVD) VRAM...")
    models_to_clear = []
    if I2V_PIPE is not None:
        models_to_clear.append(I2V_PIPE)

    clear_vram_globally(*models_to_clear)
    I2V_PIPE = None
    print("I2V (SVD) VRAM cleared.")

def generate_video_from_image(
    image_path: str,
    output_video_path: str,
    requested_num_frames: int, # This is what main_video_generator calculates based on duration/fps
    fps: int,
    width: int, 
    height: int, 
    i2v_config: I2VConfig # Contains model_native_frames
) -> str:
    pipe = load_pipeline(i2v_config)
    print(f"I2V (SVD): Requested {requested_num_frames} frames. Model will generate {i2v_config.model_native_frames} frames.")

    input_image = load_image(image_path)

    # SVD will generate its native number of frames (e.g., 14 or 25)
    # The 'num_frames' parameter to SVD pipeline is actually more like 'num_video_frames' in its docs
    # and corresponds to i2v_config.model_native_frames
    frames_to_generate_by_model = i2v_config.model_native_frames
    
    # Ensure requested frames for SVD call are at least min_request_frames if the model needs it
    # However, SVD usually generates a fixed number anyway. This is more for the 'fps' param it takes.
    # The actual number of frames generated will be model_native_frames.
    
    print(f"  Generating video from image: {image_path} (SVD will produce {frames_to_generate_by_model} frames)")

    video_frames_list = pipe( # SVD returns a list containing one tensor of frames
        input_image,
        decode_chunk_size=i2v_config.decode_chunk_size,
        num_frames=frames_to_generate_by_model, # Tell SVD to generate its standard output
        motion_bucket_id=i2v_config.motion_bucket_id,
        fps=7, # SVD (XT) is often trained with input images at 7 FPS for motion estimation. This is NOT the output FPS.
        noise_aug_strength=i2v_config.noise_aug_strength,
        height=height, 
        width=width    
    ).frames[0] # frames is a list, video data is the first element

    # video_frames_list now contains 'model_native_frames' (e.g., 25 frames)
    # We export these frames at the *target* FPS for the chunk.
    # The assemble_scene_video_from_sub_clips will handle stretching/ speeding up this chunk
    # to match the audio duration.
    export_to_video(video_frames_list, output_video_path, fps=fps) # Use the target FPS for export
    
    print(f"SVD video chunk ({len(video_frames_list)}f exported @ {fps}fps) saved to {output_video_path}")
    return output_video_path