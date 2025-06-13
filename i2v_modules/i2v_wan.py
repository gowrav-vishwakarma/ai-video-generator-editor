# In i2v_modules/i2v_wan.py
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from PIL import Image

# Import the necessary components
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel, UMT5EncoderModel, T5Tokenizer, CLIPImageProcessor

from base_modules import BaseI2V, BaseModuleConfig, ModuleCapabilities
from config_manager import DEVICE, clear_vram_globally, ContentConfig

class WanI2VConfig(BaseModuleConfig):
    """Configuration for the Wan 2.1 I2V model."""
    model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    
    num_inference_steps: int = 30
    guidance_scale: float = 5.0

class WanI2V(BaseI2V):
    """
    Image-to-Video module using the Wan 2.1 14B pipeline.
    """
    Config = WanI2VConfig

    @classmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        """Declare the capabilities of the Wan 2.1 I2V model."""
        return ModuleCapabilities(
            title="Wan 2.1 I2V (14B)",
            vram_gb_min=40.0,
            ram_gb_min=24.0,
            supported_formats=["Portrait", "Landscape"],
            supports_ip_adapter=False,
            supports_lora=False,
            max_subjects=0,
            accepts_text_prompt=True,
            accepts_negative_prompt=True
        )

    @classmethod
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Return the specific resolutions and max duration for this model."""
        return {
            "resolutions": {"base_pixel_area": 399360},  # 480P model base area
            "max_shot_duration": 4.0
        }

    def _load_pipeline(self):
        """
        Loads the WanImageToVideoPipeline following the official documentation example.
        """
        if self.pipe is not None: return

        print(f"Loading I2V pipeline ({self.config.model_id})...")

        # 1. Load individual components with appropriate dtypes
        image_encoder = CLIPVisionModel.from_pretrained(
            self.config.model_id, 
            subfolder="image_encoder", 
            torch_dtype=torch.float32
        )

        vae = AutoencoderKLWan.from_pretrained(
            self.config.model_id, 
            subfolder="vae", 
            torch_dtype=torch.float32
        )

        # 2. Create the pipeline with the components
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.config.model_id,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16
        )

        # 3. Enable model CPU offload for memory efficienc                                                                                                                                                                                                                                                                                                  y
        self.pipe.enable_model_cpu_offload()
        
        print("I2V (Wan 14B) pipeline loaded successfully.")

    def clear_vram(self):
        """Clears the VRAM used by all loaded components."""
        print(f"Clearing I2V (Wan 14B) VRAM...")
        if self.pipe is not None:
            clear_vram_globally(self.pipe)
        self.pipe = None
        print("I2V (Wan 14B) VRAM cleared.")

    def generate_video_from_image(
        self, image_path: str, output_video_path: str, target_duration: float, 
        content_config: ContentConfig, visual_prompt: str, motion_prompt: Optional[str], 
        ip_adapter_image: Optional[Union[str, List[str]]] = None
    ) -> str:
        """Generates a video by animating a source image using the 14B model."""
        self._load_pipeline()

        input_image = load_image(image_path)
        
        model_caps = self.get_model_capabilities()
        max_area = model_caps["resolutions"]["base_pixel_area"]
        aspect_ratio = input_image.height / input_image.width
        
        # Calculate dimensions using the correct scale factors
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
            num_inference_steps=self.config.num_inference_steps,
        ).frames[0]
        
        export_to_video(video_frames, output_video_path, fps=content_config.fps)
        
        print(f"Wan I2V 14B video shot saved to {output_video_path}")
        return output_video_path