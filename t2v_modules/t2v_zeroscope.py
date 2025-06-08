# In t2v_modules/t2v_zeroscope.py
import torch
from typing import Dict, Any
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

from base_modules import BaseT2V, BaseModuleConfig
from config_manager import DEVICE, clear_vram_globally

class ZeroscopeT2VConfig(BaseModuleConfig):
    model_id: str = "cerspense/zeroscope_v2_576w"
    num_inference_steps: int = 40

class ZeroscopeT2V(BaseT2V):
    Config = ZeroscopeT2VConfig
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        return {
            "resolutions": {"Portrait": (320, 576), "Landscape": (576, 320)}, # Corrected portrait
            "max_chunk_duration": 2.0 
        }
    
    def enhance_prompt(self, prompt: str, prompt_type: str = "visual") -> str:
        if prompt_type == "visual":
            return f"{prompt}, 8k, photorealistic, cinematic lighting"
        return prompt

    def _load_pipeline(self):
        if self.pipe is None:
            print(f"Loading T2V pipeline ({self.config.model_id})...")
            self.pipe = DiffusionPipeline.from_pretrained(self.config.model_id, torch_dtype=torch.float16)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.enable_model_cpu_offload()
            print(f"T2V ({self.config.model_id}) pipeline loaded.")

    def clear_vram(self):
        print(f"Clearing T2V VRAM...")
        if self.pipe is not None: clear_vram_globally(self.pipe)
        self.pipe = None
        print("T2V VRAM cleared.")

    def generate_video_from_text(
        self, prompt: str, output_video_path: str, num_frames: int, fps: int, width: int, height: int
    ) -> str:
        self._load_pipeline()
        print(f"Generating T2V ({width}x{height}) for prompt: \"{prompt[:50]}...\"")
        
        video_frames_batch = self.pipe(
            prompt=prompt, num_inference_steps=self.config.num_inference_steps,
            height=height, width=width, num_frames=num_frames, output_type="np"
        ).frames
        
        video_frames = video_frames_batch[0]
        export_to_video(video_frames, output_video_path, fps=fps)
        
        print(f"T2V video chunk saved to {output_video_path}")
        return output_video_path