import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import os
from typing import Any, Dict, List, Optional
from influencer.config import ContentConfig, MAX_VIDEO_LENGTH
import math
from moviepy import AudioFileClip

def load_t2v_pipeline(
    model_id: str,
    device: str = "cuda"
):
    """Load text-to-video pipeline (ModelScope or similar)"""
    print(f"Loading T2V pipeline: {model_id}...")
    
    pipe = DiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to(device)
    
    # Enable model CPU offload if on CUDA to save VRAM
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    
    return pipe

def generate_video_from_text(
    prompt: str,
    t2v_pipe: Any,
    output_path: str,
    fps: int = 24,
    duration: float = 3.0,
    **generation_params
) -> str:
    """Generate video directly from text prompt"""
    print(f"Generating direct video for: {prompt}")
    
    # Calculate number of frames needed based on target duration
    num_frames = int(duration * fps)  # Calculate exact frames needed
    # Ensure we have at least 8 frames
    num_frames = max(8, num_frames)
    # Limit to a reasonable maximum to prevent CUDA OOM
    num_frames = min(num_frames, 16)  # Reduced from 100 to 16
    
    # Default parameters
    default_params = {
        "num_inference_steps": 25
    }
    
    # Override with provided parameters
    params = {**default_params, **generation_params}
    
    # Generate video frames
    video_frames = t2v_pipe(
        prompt, 
        num_frames=num_frames,
        **params
    ).frames
    
    # Export to video file
    export_to_video(video_frames, output_path, fps=fps)
    
    print(f"Video saved to {output_path}")
    return output_path

def generate_scene_videos_from_text(
    visual_prompts: List[str],
    t2v_pipe: Any,
    narration_scenes: List[Dict],
    config: ContentConfig
) -> List[str]:
    """Generate videos for all scenes directly from text prompts"""
    video_paths = []
    
    for i, (prompt, scene) in enumerate(zip(visual_prompts, narration_scenes)):
        # Get model-specific max video length
        max_video_length = MAX_VIDEO_LENGTH.get(config.video_generation_mode, MAX_VIDEO_LENGTH["default"])
        
        # Check if we need to split the scene into multiple videos
        scene_duration = scene["duration"]
        
        # Calculate audio duration if an audio file exists for this scene
        audio_file = os.path.join(config.output_dir, f"scene_{i}_audio.wav")
        audio_duration = None
        if os.path.exists(audio_file):
            try:
                audio_clip = AudioFileClip(audio_file)
                audio_duration = audio_clip.duration
                audio_clip.close()
                print(f"Audio duration for scene {i}: {audio_duration}s")
            except Exception as e:
                print(f"Error getting audio duration: {e}")
        
        # Use audio duration if available, otherwise use scene duration
        target_duration = audio_duration if audio_duration is not None else scene_duration
        
        # Determine how many video segments we need
        num_segments = max(1, math.ceil(target_duration / max_video_length))
        print(f"Scene {i} needs {num_segments} video segment(s) for {target_duration}s content")
        
        scene_video_paths = []
        
        # Generate each video segment
        for segment in range(num_segments):
            # Calculate segment duration
            segment_duration = min(max_video_length, target_duration - (segment * max_video_length))
            
            if segment_duration <= 0:
                break
                
            # Generate segment video
            segment_path = os.path.join(config.output_dir, f"scene_{i}_segment_{segment}_t2v.mp4")
            generate_video_from_text(
                prompt=prompt,
                t2v_pipe=t2v_pipe,
                output_path=segment_path,
                fps=config.fps,
                duration=segment_duration,
                **config.text2vid_model_params
            )
            
            scene_video_paths.append(segment_path)
        
        # Store all video segments for this scene
        video_paths.extend(scene_video_paths)
    
    return video_paths 