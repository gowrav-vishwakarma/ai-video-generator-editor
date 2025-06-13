# In i2v_modules/i2v_slideshow.py
from typing import Dict, Any, List, Optional, Union
# --- THIS IS THE FIX: Importing ImageClip directly, matching the project's pattern ---
from moviepy.video.VideoClip import ImageClip

from base_modules import BaseI2V, BaseModuleConfig, ModuleCapabilities
from config_manager import ContentConfig

class SlideshowI2VConfig(BaseModuleConfig):
    # This module doesn't load a model, but the config is part of the contract.
    model_id: str = "moviepy_image_clip"

class SlideshowI2V(BaseI2V):
    Config = SlideshowI2VConfig

    @classmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        """
        Defines the capabilities of this simple, non-AI module.
        It uses minimal resources and doesn't support AI-specific features.
        """
        return ModuleCapabilities(
            title="Slideshow (Static Image)",
            vram_gb_min=0.1,  # Uses virtually no VRAM
            ram_gb_min=1.0,   # Uses very little RAM
            supported_formats=["Portrait", "Landscape"],
            supports_ip_adapter=False, # Not an AI model
            supports_lora=False,       # Not an AI model
            max_subjects=0,
            accepts_text_prompt=False, # Ignores prompts
            accepts_negative_prompt=False
        )

    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        This module has no native resolution and can handle long durations.
        """
        return {
            # It can handle any resolution, as it just wraps the image.
            "resolutions": {"Portrait": (1080, 1920), "Landscape": (1920, 1080)},
            "max_shot_duration": 60.0 # Can be very long
        }

    def _load_pipeline(self):
        """No pipeline to load for this module."""
        print("SlideshowI2V: No pipeline to load.")
        pass

    def clear_vram(self):
        """No VRAM to clear for this module."""
        print("SlideshowI2V: No VRAM to clear.")
        pass

    def enhance_prompt(self, prompt: str, prompt_type: str = "visual") -> str:
        """This module ignores prompts, so no enhancement is needed."""
        return prompt

    def generate_video_from_image(self, image_path: str, output_video_path: str, target_duration: float, content_config: ContentConfig, visual_prompt: str, motion_prompt: Optional[str], ip_adapter_image: Optional[Union[str, List[str]]] = None) -> str:
        """
        Creates a video by holding a static image for the target duration.
        """
        print(f"SlideshowI2V: Creating static video for {target_duration:.2f}s from {image_path}")
        
        video_clip = None
        try:
            # Create a video clip from the static image and set its duration.
            video_clip = ImageClip(image_path).with_duration(target_duration)
            
            # Use the correct syntax for write_videofile, matching video_assembly.py
            video_clip.write_videofile(
                output_video_path, 
                fps=content_config.fps,
                codec="libx264", 
                audio=False, # This is a visual-only shot
                threads=4, 
                preset="medium",
                logger=None # Suppress verbose moviepy logs
            )
            
            print(f"Slideshow video shot saved to {output_video_path}")
            return output_video_path

        except Exception as e:
            print(f"Error creating slideshow video: {e}")
            return "" # Return empty string on failure
        finally:
            # Ensure the clip resources are released
            if video_clip:
                video_clip.close()