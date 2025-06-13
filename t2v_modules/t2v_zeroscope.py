# In t2v_modules/t2v_zeroscope.py
import torch
from typing import Dict, Any, List, Optional, Union
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

from base_modules import BaseT2V, BaseModuleConfig, ModuleCapabilities
from config_manager import DEVICE, clear_vram_globally

class ZeroscopeT2VConfig(BaseModuleConfig):
    model_id: str = "cerspense/zeroscope_v2_576w"
    upscaler_model_id: str = "cerspense/zeroscope_v2_xl"
    
    num_inference_steps: int = 30
    guidance_scale: float = 9.0
    # --- START OF FIX: Add strength for the upscaling process ---
    upscaler_strength: float = 0.7 
    # --- END OF FIX ---

class ZeroscopeT2V(BaseT2V):
    Config = ZeroscopeT2VConfig

    @classmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        return ModuleCapabilities(
            title="Zeroscope, Port/Landscape, No Subject, 2 sec",
            vram_gb_min=8.0,
            ram_gb_min=12.0,
            supported_formats=["Portrait", "Landscape"],
            supports_ip_adapter=False, # Zeroscope does not support IP-Adapter
            supports_lora=False, # Zeroscope does not support LoRA loading
            max_subjects=0,
            accepts_text_prompt=True,
            accepts_negative_prompt=True
        )

    
    def __init__(self, config: ZeroscopeT2VConfig):
        super().__init__(config)
        self.upscaler_pipe = None

    @classmethod
    def get_model_capabilities(self) -> Dict[str, Any]:
        # Zeroscope has a fixed native resolution that is then upscaled
        base_resolution = (576, 320)
        return {
            "resolutions": {"Portrait": base_resolution, "Landscape": base_resolution},
            "max_shot_duration": 2.0 
        }

    def _load_pipeline(self):
        if self.pipe is None:
            print(f"Loading T2V pipeline ({self.config.model_id})...")
            self.pipe = DiffusionPipeline.from_pretrained(self.config.model_id, torch_dtype=torch.float16)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.enable_model_cpu_offload()
            print(f"T2V ({self.config.model_id}) pipeline loaded.")
            
        if self.upscaler_pipe is None:
            print(f"Loading T2V Upscaler pipeline ({self.config.upscaler_model_id})...")
            self.upscaler_pipe = DiffusionPipeline.from_pretrained(self.config.upscaler_model_id, torch_dtype=torch.float16)
            self.upscaler_pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.upscaler_pipe.scheduler.config)
            self.upscaler_pipe.enable_model_cpu_offload()
            print(f"T2V Upscaler ({self.config.upscaler_model_id}) pipeline loaded.")

    def clear_vram(self):
        print(f"Clearing T2V VRAM...")
        models_to_clear = [m for m in [self.pipe, self.upscaler_pipe] if m is not None]
        if models_to_clear: clear_vram_globally(*models_to_clear)
        self.pipe, self.upscaler_pipe = None, None
        print("T2V VRAM cleared.")

    def generate_video_from_text(
        self, prompt: str, output_video_path: str, num_frames: int, fps: int, width: int, height: int, ip_adapter_image: Optional[Union[str, List[str]]] = None
    ) -> str:
        self._load_pipeline()
        
        if ip_adapter_image:
            print("Warning: ZeroscopeT2V module received IP-Adapter image but does not currently implement its use.")

        negative_prompt = "blurry, low quality, watermark, bad anatomy, text, letters, distorted"
        
        # Note: Zeroscope generates at a fixed resolution, so we use its capabilities directly
        model_res = self.get_model_capabilities()["resolutions"]["Landscape"]
        
        print(f"Stage 1: Generating T2V ({model_res[0]}x{model_res[1]}) for prompt: \"{prompt[:70]}...\"")
        
        video_frames_tensor = self.pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            num_inference_steps=self.config.num_inference_steps,
            height=model_res[1], width=model_res[0], num_frames=num_frames,
            guidance_scale=self.config.guidance_scale, output_type="pt"
        ).frames
        
        print("Stage 2: Upscaling video to HD...")
        
        # --- START OF FIX ---
        upscaled_video_frames = self.upscaler_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            video=video_frames_tensor, # The argument is 'video', not 'image'.
            strength=self.config.upscaler_strength, # Add the strength parameter
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
        ).frames[0]
        # --- END OF FIX ---

        export_to_video(upscaled_video_frames, output_video_path, fps=fps)
        
        print(f"High-quality T2V video shot saved to {output_video_path}")
        return output_video_path