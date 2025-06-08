# t2i_modules/t2i_sdxl.py
import os
import torch
from dataclasses import dataclass
from typing import Optional
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from config_manager import DEVICE, clear_vram_globally

@dataclass
class T2IConfig:
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_id: Optional[str] = "stabilityai/stable-diffusion-xl-refiner-1.0"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    base_denoising_end: float = 0.8 
    refiner_denoising_start: float = 0.8

T2I_PIPE, REFINER_PIPE = None, None

def get_model_capabilities() -> dict:
    """Returns the optimal settings for the SDXL model."""
    return {
        "resolutions": {
            "Portrait": (896, 1152),
            "Landscape": (1344, 768),
        },
        # As this is an image model, it doesn't dictate chunk duration.
        # We can return a reasonable default that the I2V model will use.
        "max_chunk_duration": 3.0 
    }

def load_pipeline(config: T2IConfig):
    global T2I_PIPE, REFINER_PIPE
    if T2I_PIPE is None:
        print(f"Loading T2I pipeline (SDXL): {config.model_id}...")
        T2I_PIPE = StableDiffusionXLPipeline.from_pretrained(config.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(DEVICE)
        print("SDXL Base pipeline loaded.")
        if config.refiner_id:
            print(f"Loading T2I Refiner pipeline: {config.refiner_id}...")
            REFINER_PIPE = DiffusionPipeline.from_pretrained(config.refiner_id, text_encoder_2=T2I_PIPE.text_encoder_2, vae=T2I_PIPE.vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(DEVICE)
            print("SDXL Refiner pipeline loaded.")
    return T2I_PIPE, REFINER_PIPE

def clear_t2i_vram():
    global T2I_PIPE, REFINER_PIPE
    print("Clearing T2I (SDXL) VRAM...")
    models = [m for m in [T2I_PIPE, REFINER_PIPE] if m is not None]
    if models: clear_vram_globally(*models)
    T2I_PIPE, REFINER_PIPE = None, None
    print("T2I (SDXL) VRAM cleared.")

def generate_image(prompt: str, output_path: str, width: int, height: int, t2i_config: T2IConfig) -> str:
    pipe, refiner = load_pipeline(t2i_config)
    print(f"SDXL generating image with requested resolution: {width}x{height}")
    kwargs = {"prompt": prompt, "width": width, "height": height, "num_inference_steps": t2i_config.num_inference_steps, "guidance_scale": t2i_config.guidance_scale}
    if refiner:
        kwargs["output_type"] = "latent"; kwargs["denoising_end"] = t2i_config.base_denoising_end
    
    image = pipe(**kwargs).images[0]
    if refiner:
        print("Refining image...")
        image = refiner(prompt=prompt, image=image, denoising_start=t2i_config.refiner_denoising_start, num_inference_steps=t2i_config.num_inference_steps).images[0]
    
    image.save(output_path)
    print(f"Image saved to {output_path}")
    return output_path