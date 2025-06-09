# t2i_modules/t2i_juggernaut.py
import torch
from typing import Optional, Dict, Any
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline

from base_modules import BaseT2I, BaseModuleConfig
from config_manager import DEVICE, clear_vram_globally

class JuggernautT2IConfig(BaseModuleConfig):
    model_id: str = "RunDiffusion/Juggernaut-XL-v9"
    refiner_id: Optional[str] = None
    num_inference_steps: int = 30
    guidance_scale: float = 7.5

class JuggernautT2I(BaseT2I):
    Config = JuggernautT2IConfig

    def __init__(self, config: JuggernautT2IConfig):
        super().__init__(config)
        self.refiner_pipe = None

    def get_model_capabilities(self) -> Dict[str, Any]:
        return {
            "resolutions": {"Portrait": (896, 1152), "Landscape": (1344, 768)},
            "max_chunk_duration": 3.0 
        }

    def _load_pipeline(self):
        if self.pipe is None:
            print(f"Loading T2I pipeline (Juggernaut): {self.config.model_id}...")
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.config.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            ).to(DEVICE)
            print("Juggernaut Base pipeline loaded.")
            if self.config.refiner_id:
                # This part is simplified as Juggernaut doesn't typically use a refiner
                print(f"Refiner specified but not typically used with Juggernaut, skipping load.")

    def clear_vram(self):
        print("Clearing T2I (Juggernaut) VRAM...")
        models = [m for m in [self.pipe, self.refiner_pipe] if m is not None]
        if models: clear_vram_globally(*models)
        self.pipe, self.refiner_pipe = None, None
        print("T2I (Juggernaut) VRAM cleared.")

    def generate_image(self, prompt: str, output_path: str, width: int, height: int) -> str:
        self._load_pipeline()
        print(f"Juggernaut generating image with resolution: {width}x{height}")
        
        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale
        ).images[0]
        
        image.save(output_path)
        print(f"Image saved to {output_path}")
        return output_path