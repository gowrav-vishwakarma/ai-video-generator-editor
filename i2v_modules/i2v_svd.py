# i2v_modules/i2v_svd.py
import torch
from typing import Dict, Any, Optional
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image

from base_modules import BaseI2V, BaseModuleConfig, ModuleCapabilities
from config_manager import DEVICE, clear_vram_globally, ContentConfig

class SvdI2VConfig(BaseModuleConfig):
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    decode_chunk_size: int = 8
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    model_native_frames: int = 25

class SvdI2V(BaseI2V):
    Config = SvdI2VConfig

    @classmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        return ModuleCapabilities(
            vram_gb_min=8.0,
            ram_gb_min=12.0,
            supported_formats=["Portrait", "Landscape"],
            supports_ip_adapter=True,
            supports_lora=True, # Juggernaut is a fine-tune, can easily use LoRAs
            max_subjects=2, # Can handle one or two IP adapter images
            accepts_text_prompt=False,
            accepts_negative_prompt=True
        )


    def get_model_capabilities(self) -> Dict[str, Any]:
        return {
            "resolutions": {"Portrait": (576, 1024), "Landscape": (1024, 576)},
            "max_chunk_duration": 3.0 
        }
        
    def enhance_prompt(self, prompt: str, prompt_type: str = "visual") -> str:
        # SVD doesn't use text prompts, but this shows how you could add model-specific keywords.
        # For example, for a different model you might do:
        # if prompt_type == "visual":
        #    return f"{prompt}, 8k, photorealistic, cinematic lighting"
        return prompt # Return original for SVD

    def _load_pipeline(self):
        if self.pipe is None:
            print(f"Loading I2V pipeline (SVD): {self.config.model_id}...")
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                self.config.model_id, torch_dtype=torch.float16
            )
            self.pipe.enable_model_cpu_offload()
            print("I2V (SVD) pipeline loaded.")

    def clear_vram(self):
        print("Clearing I2V (SVD) VRAM...")
        if self.pipe is not None: clear_vram_globally(self.pipe)
        self.pipe = None
        print("I2V (SVD) VRAM cleared.")

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
        svd_target_res = self.get_model_capabilities()["resolutions"]
        aspect_ratio = "Landscape" if input_image.width > input_image.height else "Portrait"
        svd_target_width, svd_target_height = svd_target_res[aspect_ratio]
        prepared_image = self._resize_and_pad(input_image, svd_target_width, svd_target_height)

        calculated_fps = max(1, round(self.config.model_native_frames / target_duration)) if target_duration > 0 else 8
        motion_bucket_id = self.config.motion_bucket_id
        if motion_prompt:
            motion_prompt_lower = motion_prompt.lower()
            if any(w in motion_prompt_lower for w in ['fast', 'quick', 'rapid', 'zoom in', 'pan right']): motion_bucket_id = min(255, motion_bucket_id + 50)
            elif any(w in motion_prompt_lower for w in ['slow', 'gentle', 'subtle', 'still']): motion_bucket_id = max(0, motion_bucket_id - 50)
            print(f"Adjusted motion_bucket_id to {motion_bucket_id} based on prompt: '{motion_prompt}'")

        video_frames = self.pipe(
            image=prepared_image, height=svd_target_height, width=svd_target_width,
            decode_chunk_size=self.config.decode_chunk_size, num_frames=self.config.model_native_frames,
            motion_bucket_id=motion_bucket_id, noise_aug_strength=self.config.noise_aug_strength,
        ).frames[0]

        export_to_video(video_frames, output_video_path, fps=calculated_fps)
        print(f"SVD video chunk saved to {output_video_path}")
        return output_video_path