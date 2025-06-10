# In t2i_modules/t2i_juggernaut.py
import torch
from typing import List, Optional, Dict, Any, Union
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from diffusers.utils import load_image

# --- NEW: Import BitsAndBytesConfig for quantization ---
from transformers import BitsAndBytesConfig

# --- Gracefully import the correct scheduler ---
try:
    from diffusers import DPMMSolverMultistepScheduler as JuggernautScheduler
    print("INFO: Loaded recommended DPMMSolverMultistepScheduler for Juggernaut.")
except ImportError:
    print("WARNING: DPMMSolverMultistepScheduler not found. Falling back to DPMSolverMultistepScheduler.")
    from diffusers import DPMSolverMultistepScheduler as JuggernautScheduler

from base_modules import BaseT2I, BaseModuleConfig, ModuleCapabilities
from config_manager import DEVICE, clear_vram_globally

class JuggernautT2IConfig(BaseModuleConfig):
    model_id: str = "RunDiffusion/Juggernaut-XL-v9"
    refiner_id: Optional[str] = None
    # --- NEW: Flag to control memory-saving quantization ---
    use_8bit_quantization: bool = True
    num_inference_steps: int = 35
    guidance_scale: float = 6.0 
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "sdxl_models"
    ip_adapter_weight_name: str = "ip-adapter_sdxl.bin"


class JuggernautT2I(BaseT2I):
    Config = JuggernautT2IConfig

    def __init__(self, config: JuggernautT2IConfig):
        super().__init__(config)
        self.refiner_pipe = None
        self._loaded_ip_adapter_count = 0
    
    @classmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        return ModuleCapabilities(
            title="Juggernaut XL v9 (Quality), 2 Subjects considered",
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
            "resolutions": {"Portrait": (832, 1216), "Landscape": (1216, 832)},
            "max_chunk_duration": 3.0 
        }

    def enhance_prompt(self, prompt: str, prompt_type: str = "visual") -> str:
        quality_keywords = "cinematic photography, hyperdetailed, (skin details:1.1), 8k, professional lighting"
        if prompt.strip().endswith(','):
            return f"{prompt} {quality_keywords}"
        else:
            return f"{prompt}, {quality_keywords}"

    def _load_pipeline(self):
        if self.pipe is None:
            if self.config.use_8bit_quantization:
                print("Loading T2I pipeline (Juggernaut) with 8-bit quantization to save VRAM...")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                # --- START OF FIX: Remove device_map and use .to(DEVICE) instead ---
                # This prevents the accelerate hook conflict when loading IP-Adapters later.
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    self.config.model_id,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                ).to(DEVICE)
                # --- END OF FIX ---
            else:
                print(f"Loading T2I pipeline (Juggernaut) in full precision to {DEVICE}...")
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    self.config.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                ).to(DEVICE)
            
            self.pipe.scheduler = JuggernautScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)
            print(f"Juggernaut pipeline configured with {JuggernautScheduler.__name__} sampler.")

            if self.config.refiner_id:
                print(f"Refiner specified but not typically used with Juggernaut, skipping load.")

    def clear_vram(self):
        print("Clearing T2I (Juggernaut) VRAM...")
        models = [m for m in [self.pipe, self.refiner_pipe] if m is not None]
        if models: clear_vram_globally(*models)
        self.pipe, self.refiner_pipe = None, None
        self._loaded_ip_adapter_count = 0
        print("T2I (Juggernaut) VRAM cleared.")

    def generate_image(self, prompt: str, negative_prompt: str, output_path: str, width: int, height: int, ip_adapter_image: Optional[Union[str, List[str]]] = None, seed: int = -1) -> str:
        self._load_pipeline()
        
        generator = None
        if seed != -1:
            print(f"Using fixed seed for generation: {seed}")
            generator = torch.Generator(device=self.pipe.device).manual_seed(seed)
        else:
            print("Using random seed for generation.")

        pipeline_kwargs = {"generator": generator} if generator else {}
        ip_images_to_load = []

        if ip_adapter_image:
            if isinstance(ip_adapter_image, str):
                ip_images_to_load = [ip_adapter_image]
            else:
                ip_images_to_load = ip_adapter_image
        
        num_ip_images = len(ip_images_to_load)

        if num_ip_images > 0:
            print(f"Juggernaut T2I: Activating IP-Adapter with {num_ip_images} character image(s).")
            if self._loaded_ip_adapter_count != num_ip_images:
                print(f"Loading {num_ip_images} IP-Adapter(s) for the pipeline...")
                if hasattr(self.pipe, "unload_ip_adapter"): self.pipe.unload_ip_adapter()
                adapter_weights = [self.config.ip_adapter_weight_name] * num_ip_images
                self.pipe.load_ip_adapter(
                    self.config.ip_adapter_repo, 
                    subfolder=self.config.ip_adapter_subfolder, 
                    weight_name=adapter_weights
                )
                self._loaded_ip_adapter_count = num_ip_images
                print(f"Successfully loaded {self._loaded_ip_adapter_count} adapters.")
            
            scales = [0.6] * num_ip_images
            self.pipe.set_ip_adapter_scale(scales) 
            ip_images = [load_image(p) for p in ip_images_to_load]
            pipeline_kwargs["ip_adapter_image"] = ip_images
        else:
            print("Juggernaut T2I: No IP-Adapter image provided.")
            if self._loaded_ip_adapter_count > 0:
                 if hasattr(self.pipe, "unload_ip_adapter"): self.pipe.unload_ip_adapter()
                 self._loaded_ip_adapter_count = 0

        enhanced_prompt = self.enhance_prompt(prompt)
        print(f"Juggernaut generating image with resolution: {width}x{height}")
        print(f"  - Prompt: '{enhanced_prompt}'")
        print(f"  - Negative: '{negative_prompt}'")

        image = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            **pipeline_kwargs
        ).images[0]
        
        image.save(output_path)
        print(f"Image saved to {output_path}")
        return output_path