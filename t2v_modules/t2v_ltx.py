# In t2v_modules/t2v_ltx.py
import torch
from typing import Dict, Any
# --- START OF FIX: Use direct, robust import paths ---
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline
from diffusers.pipelines.ltx.pipeline_ltx_latent_upsample import LTXLatentUpsamplePipeline
# --- END OF FIX ---
from diffusers.utils import export_to_video
from PIL import Image

from base_modules import BaseT2V, BaseModuleConfig
from config_manager import DEVICE, clear_vram_globally

class LtxT2VConfig(BaseModuleConfig):
    """
    Configuration for the Lightricks LTX Text-to-Video model.
    This model uses a multi-stage generation process.
    """
    model_id: str = "Lightricks/LTX-Video"
    upscaler_model_id: str = "Lightricks/ltx-spatial-upscaler"

    num_inference_steps_main: int = 30
    num_inference_steps_denoise: int = 10
    guidance_scale: float = 9.0
    denoise_strength: float = 0.4

class LtxT2V(BaseT2V):
    """
    Implements Text-to-Video generation using the Lightricks LTX model.
    Follows the official multi-stage pipeline for high-quality results.
    """
    Config = LtxT2VConfig

    def __init__(self, config: LtxT2VConfig):
        super().__init__(config)
        self.upscaler_pipe = None

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Returns the optimal final output resolutions for the LTX model."""
        return {
            "resolutions": {
                "Portrait": (512, 704), 
                "Landscape": (704, 512)
            },
            "max_chunk_duration": 4.0
        }

    def _load_pipeline(self):
        """Loads both the main condition pipeline and the latent upsampler."""
        if self.pipe is None:
            print(f"Loading T2V Condition pipeline ({self.config.model_id})...")
            self.pipe = LTXConditionPipeline.from_pretrained(
                self.config.model_id, 
                torch_dtype=torch.bfloat16
            )
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_tiling()
            print("T2V Condition pipeline loaded.")

        if self.upscaler_pipe is None:
            print(f"Loading T2V Upscaler pipeline ({self.config.upscaler_model_id})...")
            self.upscaler_pipe = LTXLatentUpsamplePipeline.from_pretrained(
                self.config.upscaler_model_id,
                vae=self.pipe.vae,
                torch_dtype=torch.bfloat16
            )
            self.upscaler_pipe.enable_model_cpu_offload()
            print("T2V Upscaler pipeline loaded.")

    def clear_vram(self):
        """Clears VRAM for both LTX pipelines."""
        print("Clearing T2V (LTX) VRAM...")
        models = [m for m in [self.pipe, self.upscaler_pipe] if m is not None]
        if models:
            clear_vram_globally(*models)
        self.pipe, self.upscaler_pipe = None, None
        print("T2V (LTX) VRAM cleared.")

    def _round_to_vae_multiple(self, height: int, width: int) -> tuple[int, int]:
        """Rounds dimensions down to the nearest multiple of the VAE's compression ratio."""
        ratio = self.pipe.vae_spatial_compression_ratio
        height = height - (height % ratio)
        width = width - (width % ratio)
        return height, width

    def generate_video_from_text(
        self, prompt: str, output_video_path: str, num_frames: int, fps: int, width: int, height: int
    ) -> str:
        self._load_pipeline()
        
        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted, text, watermark"
        downscale_factor = 2 / 3
        
        print("Stage 1: Generating low-resolution latents...")
        downscaled_height, downscaled_width = int(height * downscale_factor), int(width * downscale_factor)
        downscaled_height, downscaled_width = self._round_to_vae_multiple(downscaled_height, downscaled_width)
        
        latents = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=self.config.num_inference_steps_main,
            guidance_scale=self.config.guidance_scale,
            output_type="latent",
        ).frames

        print("Stage 2: Upscaling latents...")
        upscaled_width, upscaled_height = downscaled_width * 2, downscaled_height * 2
        
        upscaled_latents = self.upscaler_pipe(
            latents=latents,
            output_type="latent"
        ).frames

        print("Stage 3: Denoising upscaled latents...")
        video_frames = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=upscaled_width,
            height=upscaled_height,
            num_frames=num_frames,
            denoise_strength=self.config.denoise_strength,
            num_inference_steps=self.config.num_inference_steps_denoise,
            latents=upscaled_latents,
            guidance_scale=self.config.guidance_scale,
            output_type="pil",
        ).frames[0]

        print(f"Stage 4: Resizing frames to {width}x{height}...")
        final_video = [frame.resize((width, height), resample=Image.LANCZOS) for frame in video_frames]

        export_to_video(final_video, output_video_path, fps=fps)
        
        print(f"High-quality LTX T2V video chunk saved to {output_video_path}")
        return output_video_path