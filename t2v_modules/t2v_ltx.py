# In t2v_modules/t2v_ltx.py
import torch
from typing import Dict, Any

# --- Import the necessary pipelines and configs for quantization ---
from diffusers import LTXPipeline, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
# We need both transformers' and diffusers' BitsAndBytesConfig for this specific model
from transformers import T5EncoderModel, BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig


from base_modules import BaseT2V, BaseModuleConfig, ModuleCapabilities
from config_manager import DEVICE, clear_vram_globally

class LtxT2VConfig(BaseModuleConfig):
    """
    Configuration for the Lightricks LTX Text-to-Video model.
    Includes a flag to enable memory-saving 8-bit quantization.
    """
    model_id: str = "Lightricks/LTX-Video" 
    use_8bit_quantization: bool = True
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    decode_timestep: float = 0.03
    decode_noise_scale: float = 0.025

class LtxT2V(BaseT2V):
    """
    Implements Text-to-Video generation using the LTXPipeline,
    with support for 8-bit quantization to save VRAM.
    """
    Config = LtxT2VConfig

    @classmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        return ModuleCapabilities(
            vram_gb_min=8.0,
            ram_gb_min=12.0,
            supported_formats=["Portrait", "Landscape"],
            supports_ip_adapter=True,
            supports_lora=True, # Juggernaut is a fine-tune, can easily use LoRAs
            max_subjects=2, # Can handle one or two IP adapter images
            accepts_text_prompt=True,
            accepts_negative_prompt=True
        )

    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Returns the optimal final output resolutions."""
        return {
            "resolutions": {
                "Portrait": (512, 768), 
                "Landscape": (768, 512)
            },
            "max_chunk_duration": 5.0
        }

    def _load_pipeline(self):
        """
        Loads the LTX pipeline. If quantization is enabled, it loads the
        heavy components in 8-bit to save significant VRAM.
        """
        if self.pipe is not None:
            return

        if self.config.use_8bit_quantization:
            print(f"Loading T2V pipeline ({self.config.model_id}) with 8-bit quantization...")

            # 1. Define the quantization config for the T5 text_encoder (from transformers)
            text_encoder_quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True)

            # 2. Load the heavy text encoder with its quantization config
            text_encoder_8bit = T5EncoderModel.from_pretrained(
                self.config.model_id,
                subfolder="text_encoder",
                quantization_config=text_encoder_quant_config,
                torch_dtype=torch.float16, # It's good practice to set dtype even when quantizing
            )

            # 3. Define the quantization config for the Transformer (from diffusers)
            transformer_quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)

            # 4. Load the even heavier video transformer with its quantization config
            transformer_8bit = LTXVideoTransformer3DModel.from_pretrained(
                self.config.model_id,
                subfolder="transformer",
                quantization_config=transformer_quant_config,
                torch_dtype=torch.float16,
            )

            # 5. Assemble the final pipeline from the quantized components.
            #    As per the traceback and docs, this pipeline requires 'balanced' for device_map
            #    when loading components manually.
            self.pipe = LTXPipeline.from_pretrained(
                self.config.model_id,
                text_encoder=text_encoder_8bit,
                transformer=transformer_8bit,
                torch_dtype=torch.float16,
                device_map="balanced", # <-- THE FIX: Use 'balanced' instead of 'auto'
            )
            print("Quantized T2V pipeline loaded successfully with 'balanced' device map.")

        else:
            # Original, non-quantized loading path
            print(f"Loading T2V pipeline ({self.config.model_id}) in full precision...")
            self.pipe = LTXPipeline.from_pretrained(
                self.config.model_id, 
                torch_dtype=torch.bfloat16 # Use bfloat16 for full precision as recommended
            )
            # enable_model_cpu_offload is an alternative for non-quantized loading
            self.pipe.enable_model_cpu_offload()

        # VAE tiling is beneficial in both cases for memory efficiency during decoding
        self.pipe.vae.enable_tiling()
        print("VAE tiling enabled for memory efficiency.")

    def clear_vram(self):
        """Clears the VRAM used by the LTX pipeline."""
        print(f"Clearing T2V (LTX) VRAM...")
        if self.pipe is not None: 
            # When using device_map, accelerate handles the model parts.
            # Setting to None and calling gc is the main way to clear.
            clear_vram_globally(self.pipe, self.pipe.text_encoder, self.pipe.transformer)
        self.pipe = None
        print("T2V (LTX) VRAM cleared.")

    def generate_video_from_text(
        self, prompt: str, output_video_path: str, num_frames: int, fps: int, width: int, height: int
    ) -> str:
        """Generates a video directly from a text prompt using the loaded LTXPipeline."""
        self._load_pipeline()
        
        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted, text, watermark, bad anatomy"
        
        print(f"Generating LTX T2V ({width}x{height}) for prompt: \"{prompt[:50]}...\"")
        
        # The LTX pipeline call is correct and doesn't need changes
        video_frames = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            decode_timestep=self.config.decode_timestep,
            decode_noise_scale=self.config.decode_noise_scale
        ).frames[0]
        
        export_to_video(video_frames, output_video_path, fps=fps)
        
        print(f"LTX T2V video chunk saved to {output_video_path}")
        return output_video_path