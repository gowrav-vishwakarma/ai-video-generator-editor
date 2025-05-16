import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
import os
from typing import Any, Dict, List, Optional
from PIL import Image
from influencer.config import ContentConfig

def load_i2v_pipeline(
    model_id: str,
    device: str = "cuda"
):
    """Load image-to-video pipeline (SVD or similar)"""
    print(f"Loading I2V pipeline: {model_id}...")
    
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to(device)
    
    # Enable model CPU offload if on CUDA to save VRAM
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    
    return pipe

def generate_video_from_image(
    image: Image.Image,
    i2v_pipe: Any,
    output_path: str,
    fps: int = 24,
    duration: float = 3.0,
    **generation_params
) -> str:
    """Generate video from input image"""
    print(f"Generating video from image using SVD...")
    
    # Calculate number of frames needed based on target duration
    num_frames = min(int(duration * fps), 25)  # Limit to 25 frames max
    # Ensure we have at least 8 frames (minimum for SVD)
    num_frames = max(8, num_frames)
    
    # Default parameters
    default_params = {
        "decode_chunk_size": 4,
        "motion_bucket_id": 127,  # Adjust for more/less motion
        "noise_aug_strength": 0.02  # Default value
    }
    
    # Override with provided parameters
    params = {**default_params, **generation_params}
    
    # Generate video frames
    video_frames = i2v_pipe(
        image,
        num_frames=num_frames,
        fps=fps,
        **params
    ).frames[0]
    
    # Export to video file
    export_to_video(video_frames, output_path, fps=fps)
    
    print(f"Video saved to {output_path}")
    return output_path

def generate_scene_videos_from_images(
    image_paths: List[str],
    i2v_pipe: Any,
    narration_scenes: List[Dict],
    config: ContentConfig
) -> List[str]:
    """Generate videos for all scenes from keyframe images"""
    video_paths = []
    
    for i, (image_path, scene) in enumerate(zip(image_paths, narration_scenes)):
        # Load image
        image = Image.open(image_path)
        
        # Generate video
        video_path = os.path.join(config.output_dir, f"scene_{i}_svd.mp4")
        generate_video_from_image(
            image=image,
            i2v_pipe=i2v_pipe,
            output_path=video_path,
            fps=config.fps,
            duration=scene["duration"],
            **config.img2vid_model_params
        )
        
        video_paths.append(video_path)
    
    return video_paths 