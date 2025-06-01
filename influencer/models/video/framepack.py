import torch
from typing import List, Optional, Union, Dict, Any
import os
import sys
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers.utils import export_to_video
from transformers import SiglipImageProcessor, SiglipVisionModel
from influencer.config import ContentConfig


# Import from FramePack-Studio's diffusers_helper
from diffusers_helper.hunyuan import encode_prompt_conds, vae_encode, vae_decode
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, get_gpu_device, get_cuda_free_memory_gb, load_model_as_complete, move_model_to_device_with_memory_preservation
from diffusers_helper.memory import offload_model_from_device_for_memory_preservation, DynamicSwapInstaller, unload_complete_models
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop
from diffusers_helper.clip_vision import hf_clip_vision_encode

# Global state for model management
text_encoder = None
text_encoder_2 = None
tokenizer = None
tokenizer_2 = None
vae = None
feature_extractor = None
image_encoder = None
transformer = None
high_vram = None

# Configure PyTorch CUDA memory management
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:512,"  # Maximum size of a single memory block
        "expandable_segments:True"  # Allow memory segments to expand
    )

# Add memory reservation constant
RESERVED_VRAM_GB = 6.0  # Reserve 6GB VRAM as recommended by FramePack-Batch
HIGH_VRAM_THRESHOLD = 24.0  # More reasonable threshold for high VRAM mode

def get_available_vram():
    """Get available VRAM after reservation"""
    if torch.cuda.is_available():
        device = get_gpu_device()
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        free_memory = get_cuda_free_memory_gb(device)
        # Ensure we maintain the reserved memory
        available_memory = max(0, free_memory - RESERVED_VRAM_GB)
        print(f"Total VRAM: {total_memory:.2f}GB, Free: {free_memory:.2f}GB, Available: {available_memory:.2f}GB (Reserved: {RESERVED_VRAM_GB}GB)")
        return available_memory
    return 0

def move_to_device_with_memory_preservation(model, device='cuda', preserved_memory_gb=8):
    """Move model to device while preserving specified amount of memory"""
    print(f'Moving {model.__class__.__name__} to {device} with preserved memory: {preserved_memory_gb} GB')
    
    if device == 'cuda' and torch.cuda.is_available():
        for m in model.modules():
            if get_cuda_free_memory_gb() <= preserved_memory_gb:
                torch.cuda.empty_cache()
                return
            if hasattr(m, 'weight'):
                m.to(device=device)
        model.to(device=device)
        torch.cuda.empty_cache()
    else:
        model.to(device)
    return model

def unload_models(*models):
    """Unload models from GPU memory"""
    for model in models:
        if model is not None:
            model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_framepack_pipeline(
    transformer_model_id: str = "lllyasviel/FramePack_F1_I2V_HY_20250503",
    hunyuan_model_id: str = "hunyuanvideo-community/HunyuanVideo",
    image_encoder_id: str = "lllyasviel/flux_redux_bfl",
    device: str = "cuda"
):
    """Load Hunyuan Video Framepack pipeline for image-to-video generation"""
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, transformer, high_vram
    
    print(f"Loading Framepack pipeline: transformer={transformer_model_id}, base={hunyuan_model_id}")
    
    # Clear CUDA cache first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Reserve VRAM by allocating a tensor
        reserve_tensor = torch.zeros((int(RESERVED_VRAM_GB * 1024**3 / 4),), device='cuda', dtype=torch.float32)
    
    # Check available memory and set mode
    available_mem_gb = get_available_vram()
    high_vram = available_mem_gb > HIGH_VRAM_THRESHOLD
    print(f"Available VRAM after reservation: {available_mem_gb:.2f}GB, High VRAM mode: {high_vram}")
    
    try:
        # Load all components in CPU first if not already loaded
        if text_encoder is None:
            print("Loading text encoders...")
            text_encoder = LlamaModel.from_pretrained(
                hunyuan_model_id, 
                subfolder='text_encoder', 
                torch_dtype=torch.float16,
                device_map='cpu'
            )
            text_encoder_2 = CLIPTextModel.from_pretrained(
                hunyuan_model_id, 
                subfolder='text_encoder_2', 
                torch_dtype=torch.float16,
                device_map='cpu'
            )
            
            print("Loading tokenizers...")
            tokenizer = LlamaTokenizerFast.from_pretrained(
                hunyuan_model_id, 
                subfolder='tokenizer'
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                hunyuan_model_id, 
                subfolder='tokenizer_2'
            )
            
            print("Loading VAE...")
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                hunyuan_model_id, 
                subfolder='vae', 
                torch_dtype=torch.float16,
                device_map='cpu'
            )
            
            print("Loading image encoder...")
            feature_extractor = SiglipImageProcessor.from_pretrained(
                image_encoder_id, 
                subfolder='feature_extractor'
            )
            image_encoder = SiglipVisionModel.from_pretrained(
                image_encoder_id, 
                subfolder='image_encoder', 
                torch_dtype=torch.float16,
                device_map='cpu'
            )
            
            print("Loading transformer...")
            transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                transformer_model_id,
                torch_dtype=torch.bfloat16,
                device_map='cpu'
            )
            
            # Set model configurations
            vae.eval()
            text_encoder.eval()
            text_encoder_2.eval()
            image_encoder.eval()
            transformer.eval()
            
            # Enable memory optimizations
            vae.enable_slicing()
            vae.enable_tiling()
            
            # Configure transformer
            transformer.high_quality_fp32_output_for_inference = True
            transformer.initialize_teacache(enable_teacache=True, num_steps=25)
        
        # Move models to device with proper memory management
        print("Moving models to device...")
        if high_vram:
            # High VRAM mode - load everything as complete
            load_model_as_complete(text_encoder, device, unload=True)
            load_model_as_complete(text_encoder_2, device, unload=False)
            load_model_as_complete(image_encoder, device, unload=False)
            load_model_as_complete(vae, device, unload=False)
            load_model_as_complete(transformer, device, unload=False)
        else:
            # Low VRAM mode - use DynamicSwapInstaller
            print("Operating in low VRAM mode...")
            DynamicSwapInstaller.install_model(text_encoder, device=device)
            DynamicSwapInstaller.install_model(text_encoder_2, device=device)
            DynamicSwapInstaller.install_model(image_encoder, device=device)
            DynamicSwapInstaller.install_model(vae, device=device)
            DynamicSwapInstaller.install_model(transformer, device=device)
            
            # Initial offload to preserve memory
            offload_model_from_device_for_memory_preservation(text_encoder, target_device=device, preserved_memory_gb=8)
            offload_model_from_device_for_memory_preservation(text_encoder_2, target_device=device, preserved_memory_gb=8)
            offload_model_from_device_for_memory_preservation(image_encoder, target_device=device, preserved_memory_gb=8)
            offload_model_from_device_for_memory_preservation(vae, target_device=device, preserved_memory_gb=8)
            offload_model_from_device_for_memory_preservation(transformer, target_device=device, preserved_memory_gb=8)
            
            # Clear cache after setup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"Warning: Failed to move models to device: {e}")
        print("Falling back to CPU-only mode...")
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        device = 'cpu'
    
    return {
        'text_encoder': text_encoder,
        'text_encoder_2': text_encoder_2,
        'tokenizer': tokenizer,
        'tokenizer_2': tokenizer_2,
        'vae': vae,
        'feature_extractor': feature_extractor,
        'image_encoder': image_encoder,
        'transformer': transformer,
        'device': device,
        'high_vram': high_vram
    }

def generate_video_from_image_with_framepack(
    pipe: Dict[str, Any],
    image: Union[Image.Image, str],
    prompt: str = "",
    last_image: Optional[Union[Image.Image, str]] = None,
    num_frames: int = 91,
    fps: int = 30,
    height: Optional[int] = None,
    width: Optional[int] = None,
    output_path: Optional[str] = None,
    sampling_type: str = "vanilla",
    **generation_params
) -> List[List[Image.Image]]:
    """Generate video using Hunyuan Video Framepack pipeline"""
    print(f"Generating video with Framepack from image prompt: {prompt}")
    
    # Get device and VRAM mode from pipe
    device = pipe.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    high_vram = pipe.get('high_vram', False)
    print(f"Using device: {device}, High VRAM mode: {high_vram}")
    
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
    
    # Default parameters with conservative settings
    default_params = {
        "num_inference_steps": 25,
        "guidance_scale": 10.0,
        "decode_chunk_size": 2 if high_vram else 1,
        "generator": torch.Generator(device=device).manual_seed(42)
    }
    
    # Override with provided parameters
    params = {**default_params, **generation_params}
    
    try:
        # Check available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            available_mem = get_available_vram()
            if available_mem < 2.0:  # Minimum 2GB required (after reservation)
                print("Warning: Very low VRAM available. Attempting to free memory...")
                unload_complete_models()
                available_mem = get_available_vram()
                if available_mem < 2.0:
                    print("Still insufficient memory. Switching to CPU...")
                    device = 'cpu'
                    for model_name in ['vae', 'transformer', 'text_encoder', 'text_encoder_2', 'image_encoder']:
                        if model_name in pipe:
                            pipe[model_name] = pipe[model_name].to('cpu')
        
        # Process in smaller chunks to manage memory better
        frames = []
        total_frames = num_frames
        chunk_size = 16 if high_vram else 8  # Process fewer frames at once in low VRAM mode
        
        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            print(f"Processing frames {chunk_start} to {chunk_end}...")
            
            # Check memory before each major operation
            if device == 'cuda':
                available_mem = get_available_vram()
                print(f"Available VRAM before processing chunk: {available_mem:.2f}GB")
            
            # Encode prompt for this chunk
            print("Encoding prompt...")
            if not high_vram:
                unload_complete_models()  # Unload any complete models
                move_model_to_device_with_memory_preservation(pipe['text_encoder'], device)
                move_model_to_device_with_memory_preservation(pipe['text_encoder_2'], device)
            else:
                load_model_as_complete(pipe['text_encoder'], device)
                load_model_as_complete(pipe['text_encoder_2'], device)
            
            llama_vec, clip_pooler = encode_prompt_conds(
                prompt,
                pipe['text_encoder'],
                pipe['text_encoder_2'],
                pipe['tokenizer'],
                pipe['tokenizer_2']
            )
            
            # Encode negative prompt (empty string)
            llama_vec_n, clip_pooler_n = encode_prompt_conds(
                "",
                pipe['text_encoder'],
                pipe['text_encoder_2'],
                pipe['tokenizer'],
                pipe['tokenizer_2']
            )
            
            if not high_vram:
                offload_model_from_device_for_memory_preservation(pipe['text_encoder'], target_device=device)
                offload_model_from_device_for_memory_preservation(pipe['text_encoder_2'], target_device=device)
            
            # Process input image
            print("Processing input image...")
            if not high_vram:
                torch.cuda.empty_cache()
            
            image_np = torch.from_numpy(image).float() / 127.5 - 1
            image_pt = image_np.permute(2, 0, 1)[None, :, None].to(device)
            
            # VAE encode
            print("VAE encoding...")
            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(pipe['vae'], device)
            else:
                load_model_as_complete(pipe['vae'], device)
            
            start_latent = vae_encode(image_pt, pipe['vae'])
            
            if not high_vram:
                offload_model_from_device_for_memory_preservation(pipe['vae'], target_device=device)
            
            # CLIP Vision encode
            print("CLIP Vision encoding...")
            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(pipe['image_encoder'], device)
            else:
                load_model_as_complete(pipe['image_encoder'], device)
            
            image_embeddings = hf_clip_vision_encode(
                image,
                pipe['feature_extractor'],
                pipe['image_encoder']
            )
            
            if not high_vram:
                offload_model_from_device_for_memory_preservation(pipe['image_encoder'], target_device=device)
            
            # Setup indices for packed frames
            latent_window_size = min(33, chunk_end - chunk_start)
            indices = torch.arange(chunk_start, chunk_start + latent_window_size).unsqueeze(0).to(device)
            clean_latent_indices = indices[:, :1]
            
            # Sample frames
            print(f"Sampling frames {chunk_start} to {chunk_end}...")
            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(pipe['transformer'], device)
            else:
                load_model_as_complete(pipe['transformer'], device)
            
            chunk_output = sample_hunyuan(
                transformer=pipe['transformer'],
                sampler="unipc",
                width=width,
                height=height,
                frames=chunk_end - chunk_start,
                real_guidance_scale=1.0,
                distilled_guidance_scale=params['guidance_scale'],
                guidance_rescale=0.0,
                num_inference_steps=params['num_inference_steps'],
                generator=params['generator'],
                prompt_embeds=llama_vec,
                prompt_embeds_mask=None,
                prompt_poolers=clip_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=None,
                negative_prompt_poolers=clip_pooler_n,
                device=device,
                dtype=pipe['transformer'].dtype,
                image_embeddings=image_embeddings,
                latent_indices=indices,
                clean_latents=start_latent,
                clean_latent_indices=clean_latent_indices,
            )
            
            if not high_vram:
                offload_model_from_device_for_memory_preservation(pipe['transformer'], target_device=device)
            
            # Decode frames
            print(f"Decoding frames {chunk_start} to {chunk_end}...")
            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(pipe['vae'], device)
            else:
                load_model_as_complete(pipe['vae'], device)
            
            # Process in smaller decode chunks to save memory
            decode_chunk_size = params['decode_chunk_size']
            chunk_frames = []
            for i in range(0, chunk_output.shape[2], decode_chunk_size):
                sub_chunk = chunk_output[:, :, i:i+decode_chunk_size]
                decoded_sub_chunk = vae_decode(sub_chunk, pipe['vae'])
                chunk_frames.append(decoded_sub_chunk)
                if not high_vram:
                    torch.cuda.empty_cache()
            
            chunk_frames = torch.cat(chunk_frames, dim=2)
            frames.append(chunk_frames)
            
            # Clear memory after each chunk
            if device == 'cuda':
                if not high_vram:
                    unload_complete_models()
                torch.cuda.empty_cache()
                available_mem = get_available_vram()
                print(f"Available VRAM after chunk: {available_mem:.2f}GB")
        
        # Concatenate all chunks
        frames = torch.cat(frames, dim=2)
        
        # Save video if output path provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            save_bcthw_as_mp4(frames, output_path, fps=fps)
            print(f"Video saved to {output_path}")
        
        # Final cleanup
        if not high_vram:
            unload_complete_models()
        
        return frames
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("CUDA out of memory error. Attempting to recover...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Try again with even more conservative settings
            params["decode_chunk_size"] = 1
            params["num_inference_steps"] = 20
            # Unload all models before retrying
            unload_complete_models()
            return generate_video_from_image_with_framepack(
                pipe, image, prompt, last_image, num_frames, fps,
                height, width, output_path, sampling_type, **params
            )
        else:
            raise e

def generate_scene_videos_with_framepack(
    visual_prompts: List[str],
    image_paths: List[str],
    pipe: Dict[str, Any],
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
            last_image=last_image,
            num_frames=num_frames,
            fps=config.fps,
            output_path=video_path,
            # Use framepack model parameters from config if specified
            **(config.framepack_model_params if hasattr(config, "framepack_model_params") else {})
        )
        
        video_paths.append(video_path)
    
    return video_paths 