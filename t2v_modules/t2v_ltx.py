# In t2v_modules/t2v_ltx.py
import torch
from typing import Dict, Any, List, Optional, Union
import os

# --- Import the necessary pipelines and configs ---
from diffusers import LTXPipeline, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5EncoderModel, BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

from base_modules import BaseT2V, BaseModuleConfig, ModuleCapabilities
from config_manager import DEVICE, clear_vram_globally

class LtxT2VConfig(BaseModuleConfig):
    model_id: str = "Lightricks/LTX-Video" 
    use_8bit_quantization: bool = True
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    decode_timestep: float = 0.03
    decode_noise_scale: float = 0.025
    # No IP-Adapter configs needed as this pipeline doesn't support them

class LtxT2V(BaseT2V):
    Config = LtxT2VConfig

    # No __init__ needed if we just have the default behavior

    @classmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        """This module is for pure T2V and does NOT support IP-Adapters."""
        return ModuleCapabilities(
            title="LTX, Port/Landscape, No Subject, 5 sec",
            vram_gb_min=8.0,
            ram_gb_min=12.0,
            supported_formats=["Portrait", "Landscape"],
            # --- THE CRITICAL CHANGE: Be honest about capabilities ---
            supports_ip_adapter=False,
            supports_lora=False, # This pipeline doesn't have a LoRA loader either
            max_subjects=0, 
            accepts_text_prompt=True,
            accepts_negative_prompt=True
        )
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        return {"resolutions": {"Portrait": (512, 768), "Landscape": (768, 512)}, "max_shot_duration": 5.0}

    def _load_pipeline(self):
        if self.pipe is not None: return

        if self.config.use_8bit_quantization:
            print(f"Loading T2V pipeline ({self.config.model_id}) with 8-bit quantization...")
            text_encoder_quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True)
            text_encoder_8bit = T5EncoderModel.from_pretrained(self.config.model_id, subfolder="text_encoder", quantization_config=text_encoder_quant_config, torch_dtype=torch.float16)
            transformer_quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
            transformer_8bit = LTXVideoTransformer3DModel.from_pretrained(self.config.model_id, subfolder="transformer", quantization_config=transformer_quant_config, torch_dtype=torch.float16)
            
            # Note: We are no longer passing the `image_encoder` as it was being ignored.
            self.pipe = LTXPipeline.from_pretrained(
                self.config.model_id,
                text_encoder=text_encoder_8bit,
                transformer=transformer_8bit,
                torch_dtype=torch.float16,
                device_map="balanced",
            )
            print("Quantized T2V pipeline loaded successfully.")
        else:
            print(f"Loading T2V pipeline ({self.config.model_id}) in full precision...")
            self.pipe = LTXPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.bfloat16
            )
            self.pipe.enable_model_cpu_offload()

        self.pipe.vae.enable_tiling()
        print("VAE tiling enabled for memory efficiency.")

    def clear_vram(self):
        print(f"Clearing T2V (LTX) VRAM...")
        if self.pipe is not None:
            clear_vram_globally(self.pipe)
        self.pipe = None
        print("T2V (LTX) VRAM cleared.")

    def generate_video_from_text(
        self, prompt: str, output_video_path: str, num_frames: int, fps: int, width: int, height: int, ip_adapter_image: Optional[Union[str, List[str]]] = None
    ) -> str:
        self._load_pipeline()

        # --- THE GRACEFUL HANDLING ---
        # If character images are passed, inform the user they are being ignored.
        if ip_adapter_image:
            print("="*50)
            print("WARNING: The LtxT2V module does not support IP-Adapters for character consistency.")
            print("The provided character images will be ignored for this T2V generation.")
            print("="*50)
        
        # All IP-Adapter logic is removed. We just call the pipeline.
        pipeline_kwargs = {}
        
        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted, text, watermark, bad anatomy"
        print(f"Generating LTX T2V ({width}x{height}) for prompt: \"{prompt[:50]}...\"")
        
        video_frames = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            decode_timestep=self.config.decode_timestep,
            decode_noise_scale=self.config.decode_noise_scale,
            **pipeline_kwargs
        ).frames[0]
        
        export_to_video(video_frames, output_video_path, fps=fps)
        
        print(f"LTX T2V video shot saved to {output_video_path}")
        return output_video_path