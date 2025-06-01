import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
import os
from typing import Any, Dict, List, Optional
from PIL import Image
from influencer.config import ContentConfig, MAX_VIDEO_LENGTH
import math
from moviepy import AudioFileClip, VideoFileClip, CompositeVideoClip
import gc
import numpy as np

def clear_vram(*models_or_pipelines):
    """Clear VRAM by moving models to CPU and collecting garbage"""
    for item in models_or_pipelines:
        if hasattr(item, 'cpu') and callable(getattr(item, 'cpu')):
            item.cpu()
        elif hasattr(item, 'model') and hasattr(item.model, 'cpu'):
            item.model.cpu()
    del models_or_pipelines
    torch.cuda.empty_cache()
    gc.collect()
    print("VRAM cleared and memory collected.")

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
    num_frames = int(duration * fps)  # Calculate exact frames needed
    # Ensure we have at least 8 frames (minimum for SVD)
    num_frames = max(8, num_frames)
    # Limit to a reasonable maximum to prevent CUDA OOM
    num_frames = min(num_frames, 16)  # Reduced from 100 to 16
    
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
        
        # Get scene duration from narration or audio
        scene_duration = scene["duration"]
        
        # Calculate audio duration if an audio file exists for this scene
        audio_file = os.path.join(config.output_dir, f"scene_{i}_audio.wav")
        audio_duration = None
        if os.path.exists(audio_file):
            try:
                audio_clip = AudioFileClip(audio_file)
                audio_duration = audio_clip.duration
                audio_clip.close()
            except Exception as e:
                print(f"Warning: Could not get audio duration: {e}")
        
        # Use audio duration if available, otherwise use scene duration
        target_duration = audio_duration if audio_duration is not None else scene_duration
        print(f"Audio duration for scene {i}: {target_duration}s")
        
        # Generate video at a lower FPS to get more frames
        generation_fps = 8  # Generate at 8 FPS
        num_frames = min(16, int(target_duration * generation_fps))  # Max 16 frames
        num_frames = max(8, num_frames)  # Min 8 frames
        
        print(f"Generating {num_frames} frames at {generation_fps} FPS...")
        
        # Clear VRAM before generation
        torch.cuda.empty_cache()
        gc.collect()
        
        # Generate with maximum motion
        video_frames = i2v_pipe(
            image,
            num_frames=num_frames,
            fps=generation_fps,  # Lower FPS during generation
            motion_bucket_id=127,  # Higher motion for more dynamic movement
            noise_aug_strength=0.02,  # Default noise
            decode_chunk_size=4  # Reduced chunk size for RTX 4090
        ).frames[0]
        
        # Save initial video
        temp_path = os.path.join(config.output_dir, f"scene_{i}_temp.mp4")
        export_to_video(video_frames, temp_path, fps=generation_fps)
        
        # Load and adjust speed to match target duration
        clip = VideoFileClip(temp_path)
        
        # Calculate speedup factor to match target duration
        speed_factor = clip.duration / target_duration
        
        # Speed up the clip
        final_clip = clip.speedx(speed_factor)
        
        # Save final video
        final_path = os.path.join(config.output_dir, f"scene_{i}_final.mp4")
        final_clip.write_videofile(
            final_path,
            fps=config.fps,  # Output at target FPS
            audio=False,
            codec='libx264',
            preset='medium'
        )
        
        # Clean up
        clip.close()
        final_clip.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        video_paths.append(final_path)
    
    return video_paths 