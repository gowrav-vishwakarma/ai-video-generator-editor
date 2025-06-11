# In i2v_modules/i2v_wan.py
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from PIL import Image

# --- Important: Import the specific classes and utilities for this model ---
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel, BitsAndBytesConfig as TransformersBitsAndBytesConfig

from base_modules import BaseI2V, BaseModuleConfig, ModuleCapabilities
from config_manager import DEVICE, clear_vram_globally, ContentConfig

class WanI2VConfig(BaseModuleConfig):
    """Configuration for the Wan 2.1 I2V model."""
    # This is the 14B parameter model from the official I2V documentation.
    model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    
    # --- NEW: Flag to control memory-saving quantization ---
    use_8bit_quantization: bool = True
    
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    # The I2V example doesn't use a 'strength' parameter in the final call.
    # We will omit it to match the documentation.

class WanI2V(BaseI2V):
    """
    Image-to-Video module using the Wan 2.1 14B pipeline.
    This module animates a source image based on a text prompt.
    It includes an option for 8-bit quantization to reduce VRAM usage.
    """
    Config = WanI2VConfig

    def __init__(self, config: WanI2VConfig):
        super().__init__(config)
        # We need to store the image encoder separately for this pipeline
        self.image_encoder = None
        self.vae = None

    @classmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        """Declare the capabilities of the Wan 2.1 I2V model."""
        return ModuleCapabilities(
            title="Wan 2.1 I2V (14B, Quantized, High VRAM)",
            vram_gb_min=20.0, # A realistic estimate for the 14B model with quantization
            ram_gb_min=16.0,
            supported_formats=["Portrait", "Landscape"],
            supports_ip_adapter=False,
            supports_lora=False,
            max_subjects=0,
            accepts_text_prompt=True,
            accepts_negative_prompt=True
        )

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Return the specific resolutions and max duration for this model."""
        # Using 480p as the base, which is 832*480 = 399360 pixels
        return {
            "resolutions": {"base_pixel_area": 399360}, # We'll calculate specific res later
            "max_chunk_duration": 4.0 # A safe bet for a large model
        }

    def _load_pipeline(self):
        """Loads the WanImageToVideoPipeline with optional 8-bit quantization."""
        if self.pipe is not None: return

        print(f"Loading I2V pipeline ({self.config.model_id})...")

        # The VAE is sensitive and the docs use float32. We will NOT quantize it.
        print("Loading VAE in float32 precision...")
        self.vae = AutoencoderKLWan.from_pretrained(
            self.config.model_id, subfolder="vae", torch_dtype=torch.float32
        )

        if self.config.use_8bit_quantization:
            print("Loading Image Encoder with 8-bit quantization...")
            bnb_config_transformers = TransformersBitsAndBytesConfig(load_in_8bit=True)
            self.image_encoder = CLIPVisionModel.from_pretrained(
                self.config.model_id,
                subfolder="image_encoder",
                quantization_config=bnb_config_transformers,
            )
            
            # The main pipeline contains the large DiT transformer. We load it with CPU offloading,
            # which is the recommended way to handle large models and is more stable than forced quantization
            # on the entire pipeline object.
            print("Loading main pipeline with model CPU offloading enabled...")
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                self.config.model_id,
                vae=self.vae,
                image_encoder=self.image_encoder,
                torch_dtype=torch.bfloat16 # Use bfloat16 for the parts that aren't quantized
            )
            self.pipe.enable_model_cpu_offload() # This is the key to managing VRAM

        else: # Load in full precision
            print("Loading Image Encoder in full precision...")
            self.image_encoder = CLIPVisionModel.from_pretrained(
                self.config.model_id, subfolder="image_encoder", torch_dtype=torch.float32
            )
            print("Loading main pipeline in full precision with CPU offloading...")
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                self.config.model_id,
                vae=self.vae,
                image_encoder=self.image_encoder,
                torch_dtype=torch.bfloat16
            )
            self.pipe.enable_model_cpu_offload()

        print("I2V (Wan 14B) pipeline loaded successfully.")

    def clear_vram(self):
        """Clears the VRAM used by all loaded components."""
        print(f"Clearing I2V (Wan 14B) VRAM...")
        # Clear all components
        models_to_clear = [m for m in [self.pipe, self.image_encoder, self.vae] if m is not None]
        if models_to_clear:
            clear_vram_globally(*models_to_clear)
        self.pipe, self.image_encoder, self.vae = None, None, None
        print("I2V (Wan 14B) VRAM cleared.")

    def generate_video_from_image(
        self, image_path: str, output_video_path: str, target_duration: float, 
        content_config: ContentConfig, visual_prompt: str, motion_prompt: Optional[str], 
        ip_adapter_image: Optional[Union[str, List[str]]] = None
    ) -> str:
        """Generates a video by animating a source image using the 14B model."""
        self._load_pipeline()

        input_image = load_image(image_path)
        
        # --- Replicate the exact resolution calculation logic from the documentation ---
        model_caps = self.get_model_capabilities()
        max_area = model_caps["resolutions"]["base_pixel_area"]
        aspect_ratio = input_image.height / input_image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        
        h = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        w = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        prepared_image = input_image.resize((w, h))

        num_frames = int(target_duration * content_config.fps)
        full_prompt = f"{visual_prompt}, {motion_prompt}" if motion_prompt else visual_prompt
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        print(f"Generating Wan I2V ({w}x{h}) from image: {image_path}")
        print(f"  - Prompt: \"{full_prompt[:70]}...\"")

        video_frames = self.pipe(
            image=prepared_image,
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            height=h,
            width=w,
            num_frames=num_frames,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps
        ).frames[0]
        
        export_to_video(video_frames, output_video_path, fps=content_config.fps)
        
        print(f"Wan I2V 14B video chunk saved to {output_video_path}")
        return output_video_path