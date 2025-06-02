# t2i_modules/t2i_sdxl.py
import os
import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from config_manager import T2IConfig, DEVICE, clear_vram_globally

T2I_PIPE = None
REFINER_PIPE = None # Optional

def load_pipeline(config: T2IConfig):
    global T2I_PIPE, REFINER_PIPE
    if T2I_PIPE is None:
        print(f"Loading T2I pipeline (SDXL): {config.model_id}...")
        T2I_PIPE = StableDiffusionXLPipeline.from_pretrained(
            config.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(DEVICE)
        print("SDXL Base pipeline loaded.")

        if config.refiner_id:
            print(f"Loading T2I Refiner pipeline: {config.refiner_id}...")
            REFINER_PIPE = DiffusionPipeline.from_pretrained(
                config.refiner_id, text_encoder_2=T2I_PIPE.text_encoder_2, vae=T2I_PIPE.vae,
                torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
            ).to(DEVICE)
            print("SDXL Refiner pipeline loaded.")
    return T2I_PIPE, REFINER_PIPE

def clear_t2i_vram():
    global T2I_PIPE, REFINER_PIPE
    print("Clearing T2I (SDXL) VRAM...")
    models_to_clear = []
    if T2I_PIPE is not None:
        models_to_clear.append(T2I_PIPE)
    if REFINER_PIPE is not None:
        models_to_clear.append(REFINER_PIPE)
    
    clear_vram_globally(*models_to_clear) # Use the global one
    T2I_PIPE = None
    REFINER_PIPE = None
    print("T2I (SDXL) VRAM cleared.")

def generate_image(
    prompt: str, 
    output_path: str, 
    width: int, 
    height: int, 
    t2i_config: T2IConfig
) -> str:
    pipe, refiner = load_pipeline(t2i_config) # Ensures pipelines are loaded
    print(f"Generating image ({width}x{height}) for prompt: \"{prompt[:50]}...\"")

    image_kwargs = {
        "prompt": prompt,
        "num_inference_steps": t2i_config.num_inference_steps,
        "guidance_scale": t2i_config.guidance_scale,
        "width": width,
        "height": height
    }

    if refiner:
        image_kwargs["output_type"] = "latent" # Generate latent for refiner
        # denoising_end could be a config param: t2i_config.denoising_end_base
        image_kwargs["denoising_end"] = 0.8 
    
    latents = pipe(**image_kwargs).images[0]

    if refiner:
        print("Refining image...")
        image = refiner(
            prompt=prompt,
            image=latents, # Pass latents to refiner
            # denoising_start could be a config param: t2i_config.denoising_start_refiner
            denoising_start=0.8, 
            num_inference_steps=t2i_config.num_inference_steps # Can use same or different steps
        ).images[0]
    else:
        image = latents # If no refiner, output from base is the final image

    image.save(output_path)
    print(f"Image saved to {output_path}")
    return output_path