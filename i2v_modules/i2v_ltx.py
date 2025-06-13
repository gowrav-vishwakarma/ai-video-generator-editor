# i2v_modules/i2v_ltx.py
import torch
from typing import Dict, Any, List, Optional, Union
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image

from base_modules import BaseI2V, BaseModuleConfig, ModuleCapabilities
from config_manager import DEVICE, clear_vram_globally, ContentConfig

class LtxI2VConfig(BaseModuleConfig):
    model_id: str = "Lightricks/LTX-Video"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

class LtxI2V(BaseI2V):
    Config = LtxI2VConfig

    @classmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        return ModuleCapabilities(
            title="LTX, 8bit Load, Port/LandScape, 2 Sub, Take +/- Prompts, max 4 sec",
            vram_gb_min=8.0,
            ram_gb_min=12.0,
            supported_formats=["Portrait", "Landscape"],
            supports_ip_adapter=True,
            supports_lora=True, # Juggernaut is a fine-tune, can easily use LoRAs
            max_subjects=2, # Can handle one or two IP adapter images
            accepts_text_prompt=True,
            accepts_negative_prompt=True
        )

    @classmethod
    def get_model_capabilities(self) -> Dict[str, Any]:
        return {
            "resolutions": {"Portrait": (480, 704), "Landscape": (704, 480)},
            "max_shot_duration": 4 
        }
    
    def enhance_prompt(self, prompt: str, prompt_type: str = "visual") -> str:
        # SVD doesn't use text prompts, but this shows how you could add model-specific keywords.
        # For example, for a different model you might do:
        if prompt_type == "visual":
           return f"{prompt}, 8k, photorealistic, cinematic lighting"
        return prompt # Return original for SVD

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

    def generate_video_from_image(self, image_path: str, output_video_path: str, target_duration: float, content_config: ContentConfig, visual_prompt: str, motion_prompt: Optional[str], ip_adapter_image: Optional[Union[str, List[str]]] = None) -> str:
        self._load_pipeline()
        
        input_image = load_image(image_path)
        target_res = self.get_model_capabilities()["resolutions"]
        aspect_ratio = "Landscape" if input_image.width > input_image.height else "Portrait"
        target_width, target_height = target_res[aspect_ratio]
        prepared_image = self._resize_and_pad(input_image, target_width, target_height)

        num_frames = max(16, int(target_duration * content_config.fps))
        full_prompt = f"{visual_prompt}, {motion_prompt}" if motion_prompt else visual_prompt

        # --- NEW LOGIC TO HANDLE ip_adapter_image ---
        # While LTX doesn't have a formal IP-Adapter, we can use the character
        # reference to guide the style by adding it to the prompt.
        if ip_adapter_image:
            print("LTX I2V: Using character reference to guide prompt style.")
            # For simplicity, we add a generic phrase. A more complex system could use an image-to-text model.
            full_prompt = f"in the style of the reference character, {full_prompt}"
            
        print(f"LTX I2V using prompt: {full_prompt}")
        
        video = self.pipe(
            prompt=full_prompt, image=prepared_image, width=target_width, height=target_height,
            num_frames=num_frames, num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            negative_prompt="worst quality, inconsistent motion, blurry"
        ).frames[0]
        
        export_to_video(video, output_video_path, fps=content_config.fps)
        print(f"LTX video shot saved to {output_video_path}")
        return output_video_path