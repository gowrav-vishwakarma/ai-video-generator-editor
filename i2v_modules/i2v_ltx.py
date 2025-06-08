# i2v_modules/i2v_ltx.py
import torch
from typing import Dict, Any, Optional
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image

from base_modules import BaseI2V, BaseModuleConfig
from config_manager import DEVICE, clear_vram_globally, ContentConfig

class LtxI2VConfig(BaseModuleConfig):
    model_id: str = "Lightricks/LTX-Video"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

class LtxI2V(BaseI2V):
    Config = LtxI2VConfig

    def get_model_capabilities(self) -> Dict[str, Any]:
        return {
            "resolutions": {"Portrait": (480, 704), "Landscape": (704, 480)},
            "max_chunk_duration": 2.5 
        }

    def _load_pipeline(self):
        if self.pipe is None:
            print(f"Loading I2V pipeline (LTX): {self.config.model_id}...")
            self.pipe = LTXImageToVideoPipeline.from_pretrained(self.config.model_id, torch_dtype=torch.bfloat16)
            self.pipe.enable_model_cpu_offload()
            print("I2V (LTX) pipeline loaded.")

    def clear_vram(self):
        print("Clearing I2V (LTX) VRAM...")
        if self.pipe is not None: clear_vram_globally(self.pipe)
        self.pipe = None
        print("I2V (LTX) VRAM cleared.")

    def _resize_and_pad(self, image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        original_aspect = image.width / image.height; target_aspect = target_width / target_height
        if original_aspect > target_aspect: new_width, new_height = target_width, int(target_width / original_aspect)
        else: new_height, new_width = target_height, int(target_height * original_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        background = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        background.paste(resized_image, ((target_width - new_width) // 2, (target_height - new_height) // 2))
        return background

    def generate_video_from_image(self, image_path: str, output_video_path: str, target_duration: float, content_config: ContentConfig, visual_prompt: str, motion_prompt: Optional[str]) -> str:
        self._load_pipeline()
        
        input_image = load_image(image_path)
        target_res = self.get_model_capabilities()["resolutions"]
        aspect_ratio = "Landscape" if input_image.width > input_image.height else "Portrait"
        target_width, target_height = target_res[aspect_ratio]
        prepared_image = self._resize_and_pad(input_image, target_width, target_height)

        num_frames = max(16, int(target_duration * content_config.fps))
        full_prompt = f"{visual_prompt}, {motion_prompt}" if motion_prompt else visual_prompt
        
        video = self.pipe(
            prompt=full_prompt, image=prepared_image, width=target_width, height=target_height,
            num_frames=num_frames, num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            negative_prompt="worst quality, inconsistent motion, blurry"
        ).frames[0]
        
        export_to_video(video, output_video_path, fps=content_config.fps)
        print(f"LTX video chunk saved to {output_video_path}")
        return output_video_path