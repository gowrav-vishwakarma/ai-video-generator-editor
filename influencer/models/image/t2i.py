import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
import os
from typing import Optional, Tuple, Any, Dict
from influencer.config import ContentConfig

def load_t2i_pipeline(
    model_id: str, 
    refiner_id: Optional[str] = None,
    device: str = "cuda"
):
    """Load text-to-image pipeline (SDXL or similar)"""
    print(f"Loading T2I pipeline: {model_id}...")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to(device)
    
    # Optional: Load refiner if specified
    refiner = None
    if refiner_id:
        print(f"Loading T2I refiner: {refiner_id}...")
        refiner = DiffusionPipeline.from_pretrained(
            refiner_id, 
            text_encoder_2=pipe.text_encoder_2, 
            vae=pipe.vae,
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16"
        ).to(device)
    
    return pipe, refiner

def generate_image(
    prompt: str,
    t2i_pipe: Any,
    refiner_pipe: Optional[Any] = None,
    output_path: str = None,
    **generation_params
) -> Tuple[Any, str]:
    """Generate image from text prompt"""
    print(f"Generating image for: {prompt}")
    
    # Default parameters
    default_params = {
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
    }
    
    # Override with provided parameters
    params = {**default_params, **generation_params}
    
    # Set output type for refiner pipeline
    if refiner_pipe:
        params["output_type"] = "latent"
    
    # Generate image
    result = t2i_pipe(prompt=prompt, **params)
    image = result.images[0]
    
    # Apply refiner if available
    if refiner_pipe:
        refiner_result = refiner_pipe(prompt=prompt, image=image[None, :])
        image = refiner_result.images[0]
    
    # Save image if output path provided
    if output_path:
        image.save(output_path)
        print(f"Image saved to {output_path}")
    
    return image, output_path

def generate_scene_images(
    visual_prompts: list,
    t2i_pipe: Any,
    refiner_pipe: Optional[Any],
    config: ContentConfig
) -> list:
    """Generate images for all scenes based on visual prompts"""
    image_paths = []
    
    for i, prompt in enumerate(visual_prompts):
        image_path = os.path.join(config.output_dir, f"scene_{i}_keyframe.png")
        
        # Generate image with parameters from config
        _, saved_path = generate_image(
            prompt=prompt,
            t2i_pipe=t2i_pipe,
            refiner_pipe=refiner_pipe,
            output_path=image_path,
            **config.image_model_params
        )
        
        image_paths.append(saved_path)
    
    return image_paths 