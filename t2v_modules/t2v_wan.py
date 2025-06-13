# In t2v_modules/t2v_wan.py
import torch
from typing import Dict, Any, List, Optional, Union

# --- Important: Import the specific classes for this model ---
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

from base_modules import BaseT2V, BaseModuleConfig, ModuleCapabilities
from config_manager import DEVICE, clear_vram_globally

class WanT2VConfig(BaseModuleConfig):
    """Configuration for the Wan 2.1 T2V model."""
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    # Parameters from the model card example
    num_inference_steps: int = 30 
    guidance_scale: float = 5.0

class WanT2V(BaseT2V):
    """
    Text-to-Video module using Wan 2.1 T2V 1.3B model.
    This model is efficient and produces high-quality video but does not support
    character consistency (IP-Adapter).
    """
    Config = WanT2VConfig

    @classmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        """Declare the capabilities of the Wan 2.1 model."""
        return ModuleCapabilities(
            title="Wan 2.1 (1.3B, Fast, 5s Shots)",
            vram_gb_min=15.0, # Based on the 8.19 GB requirement from the model card
            ram_gb_min=12.0,
            supported_formats=["Portrait", "Landscape"],
            # This model does not support IP-Adapter, so we are honest here.
            supports_ip_adapter=False, 
            supports_lora=False, # The pipeline does not have a LoRA loader
            max_subjects=0,
            accepts_text_prompt=True,
            accepts_negative_prompt=True
        )

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Return the specific resolutions and max duration for this model."""
        return {
            # Based on the example: width=832, height=480
            "resolutions": {"Portrait": (480, 832), "Landscape": (832, 480)},
            # Based on the example: "generate a 5-second 480P video"
            "max_shot_duration": 5.0 
        }

    def _load_pipeline(self):
        """Loads the custom WanPipeline and its required VAE."""
        if self.pipe is not None:
            return

        print(f"Loading T2V pipeline ({self.config.model_id})...")
        
        # This model requires loading the VAE separately first
        vae = AutoencoderKLWan.from_pretrained(
            self.config.model_id, 
            subfolder="vae", 
            torch_dtype=torch.float32 # VAE often works better in float32
        )

        # Then, load the main pipeline, passing the VAE to it
        self.pipe = WanPipeline.from_pretrained(
            self.config.model_id, 
            vae=vae, 
            torch_dtype=torch.bfloat16 # bfloat16 is recommended in the example
        )

        self.pipe.enable_model_cpu_offload()

        print(f"T2V ({self.config.model_id}) pipeline loaded to {DEVICE}.")

    def clear_vram(self):
        """Clears the VRAM used by the pipeline."""
        print(f"Clearing T2V (Wan) VRAM...")
        if self.pipe is not None:
            clear_vram_globally(self.pipe)
        self.pipe = None
        print("T2V (Wan) VRAM cleared.")

    def generate_video_from_text(
        self, prompt: str, output_video_path: str, num_frames: int, fps: int, width: int, height: int, ip_adapter_image: Optional[Union[str, List[str]]] = None
    ) -> str:
        """Generates a video shot using the Wan T2V pipeline."""
        self._load_pipeline()
        
        # Gracefully handle the case where character images are passed to a non-supporting model.
        if ip_adapter_image:
            print("="*50)
            print("WARNING: The WanT2V module does not support IP-Adapters for character consistency.")
            print("The provided character images will be ignored for this T2V generation.")
            print("="*50)

        # Use the detailed negative prompt from the model card for best results
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        print(f"Generating Wan T2V ({width}x{height}) for prompt: \"{prompt[:70]}...\"")
        
        video_frames = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps
        ).frames[0]
        
        # The system's config determines the final FPS, not the model's example
        export_to_video(video_frames, output_video_path, fps=fps)
        
        print(f"Wan T2V video shot saved to {output_video_path}")
        return output_video_path