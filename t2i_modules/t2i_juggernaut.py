# In t2i_modules/t2i_juggernaut.py
import torch
from typing import List, Optional, Dict, Any, Union
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from diffusers.utils import load_image

from base_modules import BaseT2I, BaseModuleConfig, ModuleCapabilities
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
    
    @classmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        return ModuleCapabilities(
            vram_gb_min=8.0,
            ram_gb_min=12.0,
            supported_formats=["Portrait", "Landscape"],
            supports_ip_adapter=True,
            supports_lora=True,
            max_subjects=2,
            accepts_text_prompt=True,
            accepts_negative_prompt=True
        )

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

    def generate_image(self, prompt: str, output_path: str, width: int, height: int, ip_adapter_image: Optional[Union[str, List[str]]] = None) -> str:
        self._load_pipeline()
        
        pipeline_kwargs = {}
        # --- MODIFIED IP-ADAPTER LOGIC with better logging ---
        print(f"Juggernaut T2I: IP-Adapter image: {ip_adapter_image}")
        if ip_adapter_image:
            print(f"Juggernaut T2I: Activating IP-Adapter with {len(ip_adapter_image)} character image(s).")
            if not hasattr(self.pipe, '_ip_adapter_loaded'):
                print("Loading IP-Adapter weights for Juggernaut pipeline...")
                self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
                self.pipe.set_ip_adapter_scale(0.7)
                self.pipe._ip_adapter_loaded = True
            
            if isinstance(ip_adapter_image, str):
                ip_images = [load_image(ip_adapter_image)]
            else:
                ip_images = [load_image(p) for p in ip_adapter_image]
            
            pipeline_kwargs["ip_adapter_image"] = ip_images
        else:
            print("Juggernaut T2I: No IP-Adapter image provided. Generating from text prompt only.")
        
        print(f"Juggernaut generating image with resolution: {width}x{height} for prompt: '{prompt}'")
        
        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            **pipeline_kwargs
        ).images[0]
        
        image.save(output_path)
        print(f"Image saved to {output_path}")
        return output_path