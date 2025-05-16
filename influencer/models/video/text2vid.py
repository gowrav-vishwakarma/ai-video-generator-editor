import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import os
from typing import Any, Dict, List, Optional
from influencer.config import ContentConfig

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
    num_frames = int(duration * fps)
    # Ensure we have enough frames
    num_frames = max(8, num_frames)
    
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
        # Generate video
        video_path = os.path.join(config.output_dir, f"scene_{i}_t2v.mp4")
        generate_video_from_text(
            prompt=prompt,
            t2v_pipe=t2v_pipe,
            output_path=video_path,
            fps=config.fps,
            duration=scene["duration"],
            **config.text2vid_model_params
        )
        
        video_paths.append(video_path)
    
    return video_paths 