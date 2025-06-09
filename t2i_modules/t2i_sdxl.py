# t2i_modules/t2i_sdxl.py
import torch
from typing import Optional, Dict, Any
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline

from base_modules import BaseT2I, BaseModuleConfig
from config_manager import DEVICE, clear_vram_globally

class SdxlT2IConfig(BaseModuleConfig):
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_id: Optional[str] = "stabilityai/stable-diffusion-xl-refiner-1.0"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    base_denoising_end: float = 0.8
    refiner_denoising_start: float = 0.8

class SdxlT2I(BaseT2I):
    Config = SdxlT2IConfig
    
    def __init__(self, config: SdxlT2IConfig):
        super().__init__(config)
        self.refiner_pipe = None

    def get_model_capabilities(self) -> Dict[str, Any]:
        return {
            "resolutions": {"Portrait": (896, 1152), "Landscape": (1344, 768)},
            "max_chunk_duration": 3.0
        }

    def _load_pipeline(self):
        if self.pipe is None:
            print(f"Loading T2I pipeline (SDXL): {self.config.model_id}...")
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.config.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            ).to(DEVICE)
            print("SDXL Base pipeline loaded.")
            if self.config.refiner_id:
                print(f"Loading T2I Refiner pipeline: {self.config.refiner_id}...")
                self.refiner_pipe = DiffusionPipeline.from_pretrained(
                    self.config.refiner_id, text_encoder_2=self.pipe.text_encoder_2,
                    vae=self.pipe.vae, torch_dtype=torch.float16,
                    use_safetensors=True, variant="fp16"
                ).to(DEVICE)
                print("SDXL Refiner pipeline loaded.")

    def clear_vram(self):
        print("Clearing T2I (SDXL) VRAM...")
        models = [m for m in [self.pipe, self.refiner_pipe] if m is not None]
        if models: clear_vram_globally(*models)
        self.pipe, self.refiner_pipe = None, None
        print("T2I (SDXL) VRAM cleared.")

    def generate_image(self, prompt: str, output_path: str, width: int, height: int) -> str:
        self._load_pipeline()
        
        kwargs = {
            "prompt": prompt, "width": width, "height": height,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale
        }
        if self.refiner_pipe:
            kwargs["output_type"] = "latent"
            kwargs["denoising_end"] = self.config.base_denoising_end

        image = self.pipe(**kwargs).images[0]
        
        if self.refiner_pipe:
            print("Refining image...")
            image = self.refiner_pipe(
                prompt=prompt, image=image,
                denoising_start=self.config.refiner_denoising_start,
                num_inference_steps=self.config.num_inference_steps
            ).images[0]
        
        image.save(output_path)
        print(f"Image saved to {output_path}")
        return output_path