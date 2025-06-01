import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

# Maximum video length constants for different models (in seconds)
MAX_VIDEO_LENGTH = {
    "img2vid": 16.0,  # Increased from 4.0 to allow longer segments like v3
    "text2vid": 4.0,  
    "framepack": 15.0,  
    "default": 16.0    # Increased default too
}

@dataclass
class ContentConfig:
    """Configuration for content generation with model selection options"""
    # Video settings
    target_video_length: float = 30.0  # Target total video length in seconds
    max_scene_length: float = 3.0      # Maximum length of each scene in seconds
    target_resolution: tuple = (1080, 1920)  # Instagram Reel 9:16
    fps: int = 24
    
    # Scene settings
    min_scenes: int = 2
    max_scenes: int = 3
    
    # Model selection
    # Text generation models
    text_model: str = "HuggingFaceH4/zephyr-7b-beta"  # LLM for script generation
    text_model_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 512,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.7
    })
    text_model_implementation: Optional[str] = None  # Custom implementation file path
    
    # Audio generation models
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"  # TTS model
    tts_model_params: Dict[str, Any] = field(default_factory=dict)
    tts_model_implementation: Optional[str] = None  # Custom implementation file path
    speaker_wav: Optional[str] = None  # For voice cloning
    
    # Image generation models
    image_model: str = "stabilityai/stable-diffusion-xl-base-1.0"  # T2I model
    image_refiner: Optional[str] = "stabilityai/stable-diffusion-xl-refiner-1.0"  # Optional refiner
    image_model_params: Dict[str, Any] = field(default_factory=lambda: {
        "num_inference_steps": 30,
        "guidance_scale": 7.5
    })
    image_model_implementation: Optional[str] = None  # Custom implementation file path
    
    # Video generation models and approach
    video_generation_mode: str = "img2vid"  # "img2vid", "text2vid", or "framepack"
    
    # For img2vid mode:
    img2vid_model: str = "stabilityai/stable-video-diffusion-img2vid-xt"  # I2V model
    img2vid_model_params: Dict[str, Any] = field(default_factory=lambda: {
        "decode_chunk_size": 4,  # Reduced from 8 to 4 for better memory usage
        "motion_bucket_id": 127,
        "noise_aug_strength": 0.02
        # Removed num_inference_steps to match v3 behavior
    })
    img2vid_model_implementation: Optional[str] = None  # Custom implementation file path
    
    # For text2vid mode:
    text2vid_model: str = "damo-vilab/text-to-video-ms-1.7b"  # T2V model
    text2vid_model_params: Dict[str, Any] = field(default_factory=lambda: {
        "num_inference_steps": 25,  # Matched with v3 version
        "num_frames": 16  # Limited to prevent OOM
    })
    text2vid_model_implementation: Optional[str] = None  # Custom implementation file path
    
    # For framepack mode:
    framepack_transformer_model: str = "lllyasviel/FramePackI2V_HY"  # Framepack transformer
    framepack_hunyuan_model: str = "hunyuanvideo-community/HunyuanVideo"  # Hunyuan base model
    framepack_image_encoder: str = "lllyasviel/flux_redux_bfl"  # Image encoder
    framepack_model_params: Dict[str, Any] = field(default_factory=lambda: {
        "num_inference_steps": 30,
        "guidance_scale": 9.0,
        "sampling_type": "inverted_anti_drifting"
    })
    framepack_model_implementation: Optional[str] = None  # Custom implementation file path
    enable_framepack_last_frame: bool = False  # Whether to use last frame control
    
    # Video assembly implementation
    video_assembly_implementation: Optional[str] = None  # Custom implementation file path
    
    # Device settings
    device: str = "cuda"  # "cuda" or "cpu"
    
    # Output settings
    output_dir: str = "instagram_content"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        # Set device based on availability
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            self.device = "cpu" 