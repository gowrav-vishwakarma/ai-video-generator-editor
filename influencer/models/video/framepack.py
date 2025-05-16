import torch
from typing import List, Optional, Union, Dict, Any
import os
from PIL import Image
from diffusers import HunyuanVideoImageToVideoPipeline as HunyuanVideoFramepackPipeline
from diffusers import HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import SiglipImageProcessor, SiglipVisionModel
from influencer.config import ContentConfig

def load_framepack_pipeline(
    transformer_model_id: str = "lllyasviel/FramePackI2V_HY",
    hunyuan_model_id: str = "hunyuanvideo-community/HunyuanVideo",
    image_encoder_id: str = "lllyasviel/flux_redux_bfl",
    device: str = "cuda"
):
    """Load Hunyuan Video Framepack pipeline for image-to-video generation"""
    print(f"Loading Framepack pipeline: transformer={transformer_model_id}, base={hunyuan_model_id}")
    
    # Define the appropriate torch data type based on hardware
    if torch.cuda.is_available() and "cuda" in device:
        # Check if BF16 is supported
        if torch.cuda.is_bf16_supported():
            transformer_dtype = torch.bfloat16
        else:
            transformer_dtype = torch.float16
    else:
        transformer_dtype = torch.float32
        
    # Load transformer model
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        transformer_model_id, 
        torch_dtype=transformer_dtype
    )
    
    # Load feature extractor and image encoder
    feature_extractor = SiglipImageProcessor.from_pretrained(
        image_encoder_id, 
        subfolder="feature_extractor"
    )
    
    image_encoder = SiglipVisionModel.from_pretrained(
        image_encoder_id, 
        subfolder="image_encoder", 
        torch_dtype=torch.float16
    )
    
    # Create the pipeline
    pipe = HunyuanVideoFramepackPipeline.from_pretrained(
        hunyuan_model_id,
        transformer=transformer,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    
    # Move to the specified device
    pipe = pipe.to(device)
    
    # Apply optimizations
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    
    return pipe

def generate_video_from_image_with_framepack(
    pipe: Any,
    image: Union[Image.Image, str],
    prompt: str = "",
    last_image: Optional[Union[Image.Image, str]] = None,
    num_frames: int = 91,
    fps: int = 30,
    height: Optional[int] = None,
    width: Optional[int] = None,
    output_path: Optional[str] = None,
    sampling_type: str = "inverted_anti_drifting",
    **generation_params
) -> List[List[Image.Image]]:
    """Generate video using Hunyuan Video Framepack pipeline"""
    print(f"Generating video with Framepack from image prompt: {prompt}")
    
    # Load the image if a path is provided
    if isinstance(image, str):
        if os.path.exists(image):
            image = Image.open(image).convert("RGB")
        else:
            raise ValueError(f"Image path does not exist: {image}")
    
    # Load the last image if provided
    if last_image is not None and isinstance(last_image, str):
        if os.path.exists(last_image):
            last_image = Image.open(last_image).convert("RGB")
        else:
            raise ValueError(f"Last image path does not exist: {last_image}")
    
    # Set height and width if not provided
    if height is None:
        height = image.height
    if width is None:
        width = image.width
    
    # Ensure height and width are multiples of 8
    height = (height // 8) * 8
    width = (width // 8) * 8
    
    # Default parameters
    default_params = {
        "num_inference_steps": 30,
        "guidance_scale": 9.0,
        "generator": torch.Generator(device=pipe.device).manual_seed(42)
    }
    
    # Override with provided parameters
    params = {**default_params, **generation_params}
    
    # Generate video
    if last_image is not None:
        print("Generating video with first and last frame control...")
        output = pipe(
            image=image,
            last_image=last_image,
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            sampling_type=sampling_type,
            **params
        ).frames[0]
    else:
        print("Generating video from single image...")
        output = pipe(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            sampling_type=sampling_type,
            **params
        ).frames[0]
    
    # Save video if output path provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        export_to_video(output, output_path, fps=fps)
        print(f"Video saved to {output_path}")
    
    return output

def generate_scene_videos_with_framepack(
    visual_prompts: List[str],
    image_paths: List[str],
    pipe: Any,
    narration_scenes: List[Dict],
    config: ContentConfig
) -> List[str]:
    """Generate videos for all scenes using Framepack pipeline"""
    video_paths = []
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Generate video for each scene
    for i, (prompt, image_path, scene) in enumerate(zip(visual_prompts, image_paths, narration_scenes)):
        # Calculate appropriate number of frames based on duration and FPS
        num_frames = int(scene["duration"] * config.fps)
        
        # Ensure minimum frame count (recommended minimum is 8 for most models)
        num_frames = max(8, num_frames)
        
        # Generate output path
        video_path = os.path.join(config.output_dir, f"scene_{i}_framepack.mp4")
        
        # Check if last frame control is enabled and we have a next image for last frame
        last_image = None
        if getattr(config, "enable_framepack_last_frame", False) and i < len(image_paths) - 1:
            # Use the next scene's image as the last frame for current scene
            # This helps create smoother transitions between scenes
            last_image = image_paths[i + 1]
            print(f"Using scene {i+1}'s image as the last frame for scene {i}")
        
        # Generate video
        generate_video_from_image_with_framepack(
            pipe=pipe,
            image=image_path,
            prompt=prompt,
            last_image=last_image,  # Pass the last image if available
            num_frames=num_frames,
            fps=config.fps,
            output_path=video_path,
            # Use framepack model parameters from config if specified
            **(config.framepack_model_params if hasattr(config, "framepack_model_params") else {})
        )
        
        video_paths.append(video_path)
    
    return video_paths 