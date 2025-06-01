#!/usr/bin/env python
# coding: utf-8

# main_script.py
import os
import sys

# Add parent directory to path to import from influencer package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and configure memory management
from influencer.utils.memory_config import configure_memory_management
configure_memory_management()

import torch
import json
from dataclasses import dataclass
from typing import List, Optional
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionXLPipeline, StableVideoDiffusionPipeline, DiffusionPipeline
from diffusers.utils import load_image, export_to_video
from TTS.api import TTS # Coqui TTS
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from moviepy.audio.AudioClip import concatenate_audioclips, AudioClip
import gc # Garbage collection
import numpy as np


@dataclass
class ContentConfig:
    """Configuration for content generation"""
    # Video settings
    target_video_length: float = 30.0  # Target total video length in seconds
    max_scene_length: float = 3.0      # Maximum length of each scene in seconds
    target_resolution: tuple = (1080, 1920)  # Instagram Reel 9:16
    fps: int = 24
    
    # Scene settings
    min_scenes: int = 2
    max_scenes: int = 3
    
    # Model settings
    use_svd_flow: bool = True  # Use SDXL -> SVD flow instead of direct T2V
    
    # Output settings
    output_dir: str = "instagram_content"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


# --- 0. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "instagram_content"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LLM paths (example, download these first)
LLM_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
# TTS model (XTTSv2 example)
TTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2" # or "tts_models/en/ljspeech/tacotron2-DDC"
# T2I model
T2I_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
T2I_REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
# I2V model (SVD)
I2V_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
# T2V model (alternative)
T2V_MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"



# --- 1. INITIALIZE MODELS (Load only when needed or keep loaded if VRAM allows) ---
def load_llm():
    print("Loading LLM...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    # Ensure pad_token is set if model doesn't have one; often same as eos_token for CausalLMs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # We won't use the pipeline directly for chat templating, but generate manually
    return model, tokenizer


def load_tts():
    print("Loading TTS model...")
    # Make sure you have the model downloaded or it will download on first run
    # For XTTSv2, you might need to specify a speaker_wav for voice cloning
    return TTS(model_name=TTS_MODEL_ID, progress_bar=True).to(DEVICE)

def load_t2i_pipeline():
    print("Loading T2I pipeline (SDXL)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        T2I_MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(DEVICE)
    # Optional: Load refiner if you want to use it
    # refiner = DiffusionPipeline.from_pretrained(
    #     T2I_REFINER_ID, text_encoder_2=pipe.text_encoder_2, vae=pipe.vae,
    #     torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    # ).to(DEVICE)
    # return pipe, refiner
    return pipe, None # Simpler for now

def load_i2v_pipeline(): # SVD
    print("Loading I2V pipeline (SVD)...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        I2V_MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    ).to(DEVICE)
    pipe.enable_model_cpu_offload() # If VRAM is tight with other models
    return pipe

def load_t2v_pipeline(): # ModelScope Text-to-Video
    print("Loading T2V pipeline (ModelScope)...")
    # Use DiffusionPipeline as per the documentation for damo-vilab/text-to-video-ms-1.7b
    pipe = DiffusionPipeline.from_pretrained(
        T2V_MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    ).to(DEVICE)
    # The .enable_model_cpu_offload() should still work if the loaded pipe supports it
    # For ModelScope, this is standard.
    pipe.enable_model_cpu_offload()
    return pipe



# --- UTILITY TO CLEAR VRAM ---
def clear_vram(*models_or_pipelines):
    for item in models_or_pipelines:
        if hasattr(item, 'cpu') and callable(getattr(item, 'cpu')):
            item.cpu() # If it's a pipeline/model with a .cpu() method
        elif hasattr(item, 'model') and hasattr(item.model, 'cpu') and callable(getattr(item.model, 'cpu')):
            item.model.cpu()
    del models_or_pipelines
    torch.cuda.empty_cache()
    gc.collect()
    print("VRAM cleared and memory collected.")



# --- 2. TEXT GENERATION ---
def generate_script_and_prompts_with_chat_template(topic, model, tokenizer, config: ContentConfig):
    print(f"Generating script and prompts for topic (chat template): {topic}")

    # Define the messages for the chat template
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant creating content for an Instagram Reel. Your response must be in valid JSON format with the following structure: {\"narration\": [{\"scene\": 1, \"text\": \"text\", \"duration\": seconds}], \"visuals\": [{\"scene\": 1, \"prompt\": \"prompt\"}], \"hashtags\": [\"tag1\", \"tag2\"]}"
        },
        {
            "role": "user",
            "content": f"""
            Create content for an Instagram Reel about "{topic}".
            The Reel should be engaging and around {config.target_video_length} seconds long.
            Each scene should be around {config.max_scene_length} seconds.
            Generate between {config.min_scenes} and {config.max_scenes} scenes.
            
            Return your response in this exact JSON format:
            {{
                "narration": [
                    {{"scene": 1, "text": "First scene narration", "duration": 3.0}},
                    {{"scene": 2, "text": "Second scene narration", "duration": 3.0}}
                ],
                "visuals": [
                    {{"scene": 1, "prompt": "Detailed visual prompt for scene 1"}},
                    {{"scene": 2, "prompt": "Detailed visual prompt for scene 2"}}
                ],
                "hashtags": ["tag1", "tag2", "tag3"]
            }}
            
            Make sure the total duration of all scenes matches approximately {config.target_video_length} seconds.
            Each visual prompt should be detailed and suitable for image/video generation.
            """
        }
    ]

    # Apply the chat template
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

    # Generate response
    generation_kwargs = {
        "input_ids": tokenized_chat,
        "max_new_tokens": 512,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.7,
        "pad_token_id": tokenizer.eos_token_id
    }

    outputs = model.generate(**generation_kwargs)
    decoded_output = tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)
    print("LLM Response (model generated part only with chat template):\n", decoded_output)

    try:
        # Try to parse the JSON response
        response_data = json.loads(decoded_output)
        
        # Extract and validate the data
        narration_scenes = []
        visual_prompts = []
        hashtags = []
        
        # Sort scenes by scene number to ensure correct order
        narration_data = sorted(response_data.get("narration", []), key=lambda x: x["scene"])
        visuals_data = sorted(response_data.get("visuals", []), key=lambda x: x["scene"])
        
        for scene in narration_data:
            narration_scenes.append({
                "text": scene["text"],
                "duration": float(scene.get("duration", config.max_scene_length))
            })
            
        for scene in visuals_data:
            visual_prompts.append(scene["prompt"])
            
        hashtags = response_data.get("hashtags", [])
        
        # Validate we have matching number of scenes
        if len(narration_scenes) != len(visual_prompts):
            raise ValueError("Mismatch between number of narration and visual scenes")
            
        # Validate scene count
        if not (config.min_scenes <= len(narration_scenes) <= config.max_scenes):
            raise ValueError(f"Scene count {len(narration_scenes)} outside allowed range [{config.min_scenes}, {config.max_scenes}]")
            
        # Validate total duration
        total_duration = sum(scene["duration"] for scene in narration_scenes)
        if abs(total_duration - config.target_video_length) > 5:  # Allow 5 second tolerance
            print(f"Warning: Total duration {total_duration}s differs from target {config.target_video_length}s")
            
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing LLM response: {e}")
        print("Using fallback content...")
        # Fallback content
        narration_scenes = [
            {"text": f"A short segment about {topic}.", "duration": config.max_scene_length},
            {"text": "Concluding thoughts on the future.", "duration": config.max_scene_length}
        ]
        visual_prompts = [
            f"Cinematic shot of {topic}, vibrant colors, high detail, trending on artstation.",
            f"Abstract representation of {topic}, thought-provoking, professional."
        ]
        hashtags = [f"#{topic.replace(' ', '')}", "#AI", "#GeneratedContent"]

    return narration_scenes, visual_prompts, hashtags



# --- 3. AUDIO GENERATION ---
def generate_audio(text, tts_model, output_path, speaker_wav=None): # speaker_wav for XTTS
    print(f"Generating audio for: {text}")
    if "xtts" in TTS_MODEL_ID.lower() and speaker_wav:
        tts_model.tts_to_file(text, speaker_wav=speaker_wav, language="en", file_path=output_path)
    else: # For other models like Tacotron
        tts_model.tts_to_file(text, file_path=output_path)
    print(f"Audio saved to {output_path}")
    return output_path



# --- 4. VISUAL GENERATION ---
# Option A: Image (SDXL) then Video (SVD) - Recommended for "Subject to Video"
def generate_image_then_video(image_prompt, i2v_pipe, t2i_pipe, refiner_pipe, scene_idx, target_duration, config: ContentConfig):
    # Generate Image (Keyframe)
    print(f"Generating keyframe image for: {image_prompt}")
    # For SDXL, you can add negative prompts, specify num_inference_steps, guidance_scale etc.
    # Using SDXL
    image = t2i_pipe(
        prompt=image_prompt,
        # negative_prompt="low quality, blurry, watermark", # Example
        num_inference_steps=30, # SDXL typically needs fewer steps
        guidance_scale=7.5,
        # If using refiner:
        # output_type="latent" if refiner_pipe else "pil",
    ).images[0]
    # if refiner_pipe:
    #     image = refiner_pipe(prompt=image_prompt, image=image[None, :]).images[0]

    image_path = os.path.join(config.output_dir, f"scene_{scene_idx}_keyframe.png")
    image.save(image_path)
    print(f"Keyframe image saved to {image_path}")

    # Calculate number of frames needed based on target duration
    num_frames = min(int(target_duration * config.fps), 16)  # Limit to 16 frames max
    # Ensure we have at least 8 frames (minimum for SVD)
    num_frames = max(8, num_frames)
    
    # Generate Video from Image (SVD)
    print(f"Generating video from image using SVD...")
    # SVD parameters
    video_frames = i2v_pipe(
        image,
        decode_chunk_size=4,  # Reduced from 8 to 4
        num_frames=num_frames,
        motion_bucket_id=127, # Adjust for more/less motion
        fps=config.fps,
        noise_aug_strength=0.02 # Default
    ).frames[0]

    video_clip_path = os.path.join(config.output_dir, f"scene_{scene_idx}_svd.mp4")
    export_to_video(video_frames, video_clip_path, fps=config.fps)
    print(f"SVD video clip saved to {video_clip_path}")
    return video_clip_path

# Option B: Direct Text-to-Video (ModelScope)
def generate_direct_video(video_prompt, t2v_pipe, scene_idx, target_duration, config: ContentConfig):
    print(f"Generating direct video for: {video_prompt}")
    # Calculate number of frames needed based on target duration
    num_frames = int(target_duration * config.fps)
    # Ensure we have at least 8 frames
    num_frames = max(8, num_frames)
    
    # ModelScope parameters
    video_frames = t2v_pipe(video_prompt, num_inference_steps=25, num_frames=num_frames).frames
    video_clip_path = os.path.join(config.output_dir, f"scene_{scene_idx}_t2v.mp4")
    export_to_video(video_frames, video_clip_path, fps=config.fps)
    print(f"T2V video clip saved to {video_clip_path}")
    return video_clip_path



# --- 5. VIDEO ASSEMBLY ---
def assemble_final_video(video_clip_paths, audio_clip_paths, narration_parts, config: ContentConfig, output_filename="final_reel.mp4"):
    print("Assembling final video...")
    final_clips = []
    source_clips_to_close = []

    # Use a specific, existing font path
    font_path_for_textclip = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    if not os.path.exists(font_path_for_textclip):
        print(f"Warning: Font file not found at: {font_path_for_textclip}")
        print("Using system default font...")
        font_path_for_textclip = None

    for i, (video_path, audio_path, narration) in enumerate(zip(video_clip_paths, audio_clip_paths, narration_parts)):
        # Load video and audio clips
        video_clip_temp = VideoFileClip(video_path)
        audio_clip_temp = AudioFileClip(audio_path)
        source_clips_to_close.extend([video_clip_temp, audio_clip_temp])

        # Get durations
        video_duration = video_clip_temp.duration
        audio_duration = audio_clip_temp.duration
        target_duration = narration["duration"]

        # Resize video to target resolution
        video_clip_resized = video_clip_temp.resized(height=config.target_resolution[1])
        if video_clip_resized.w > config.target_resolution[0]:
            video_clip_final_shape = video_clip_resized.cropped(x_center=video_clip_resized.w/2, width=config.target_resolution[0])
        else:
            video_clip_final_shape = video_clip_resized

        # Position video in center
        video_clip_positioned = video_clip_final_shape.with_position('center')

        # Handle duration mismatches
        if video_duration > target_duration:
            # If video is longer, trim it
            video_clip_timed = video_clip_positioned.subclipped(0, target_duration)
        else:
            # If video is shorter, loop it
            n_loops = int(np.ceil(target_duration / video_duration))
            video_clip_timed = concatenate_videoclips([video_clip_positioned] * n_loops)
            video_clip_timed = video_clip_timed.subclipped(0, target_duration)

        # Handle audio duration
        if audio_duration > target_duration:
            # If audio is longer, trim it
            audio_clip_timed = audio_clip_temp.subclipped(0, target_duration)
        else:
            # If audio is shorter, pad with silence
            silence_duration = target_duration - audio_duration
            # Create a silent audio clip (zero signal) with specific duration
            silence = AudioClip(frame_function=lambda t: 0, duration=silence_duration)
            audio_clip_timed = concatenate_audioclips([audio_clip_temp, silence])

        # Combine video and audio
        video_clip_with_audio = video_clip_timed.with_audio(audio_clip_timed)

        # Add text caption
        txt_clip_temp = TextClip(
            font_path_for_textclip,
            text=narration["text"],
            font_size=60,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(int(config.target_resolution[0]*0.8), None)
        )
        source_clips_to_close.append(txt_clip_temp)

        txt_clip_final = txt_clip_temp.with_position(('center', 0.8), relative=True).with_duration(target_duration)

        # Combine video and text
        scene_composite = CompositeVideoClip([video_clip_with_audio, txt_clip_final], size=config.target_resolution)
        final_clips.append(scene_composite)

    if not final_clips:
        print("No clips to assemble!")
        for clip_to_close in source_clips_to_close:
            if hasattr(clip_to_close, 'close') and callable(getattr(clip_to_close, 'close')):
                clip_to_close.close()
        return None

    # Concatenate all scenes
    final_video = concatenate_videoclips(final_clips, method="compose")
    final_video_path = os.path.join(config.output_dir, output_filename)

    try:
        final_video.write_videofile(
            final_video_path,
            fps=config.fps,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="medium",
            logger='bar'
        )
    except Exception as e:
        print(f"Error during video writing: {e}")
        print("Make sure ffmpeg is correctly installed and accessible by MoviePy.")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up all clips
        for clip_to_close in source_clips_to_close:
            if hasattr(clip_to_close, 'close') and callable(getattr(clip_to_close, 'close')):
                try:
                    clip_to_close.close()
                except Exception as e_close:
                    print(f"Error closing clip {type(clip_to_close)}: {e_close}")
        if hasattr(final_video, 'close') and callable(getattr(final_video, 'close')):
            try:
                final_video.close()
            except Exception as e_close:
                print(f"Error closing final_video: {e_close}")

    print(f"Final video saved to {final_video_path}")
    return final_video_path



# --- MAIN WORKFLOW ---
def main_automation_flow(topic, config: Optional[ContentConfig] = None):
    if config is None:
        config = ContentConfig()

    # --- Load models ---
    # Manage VRAM: Load one by one or use .cpu_offload()
    llm = None
    tts_model = None
    t2i_pipe, refiner = None, None
    i2v_pipe = None
    t2v_pipe = None
    final_video_path = None
    speaker_reference_audio = "record_out.wav" # REQUIRED for XTTS voice cloning

    try:
        # 1. LLM for Script and Prompts
        llm_model, llm_tokenizer = load_llm()
        narration_scenes, visual_prompts_scenes, hashtags = generate_script_and_prompts_with_chat_template(
            topic, llm_model, llm_tokenizer, config
        )
        clear_vram(llm_model)

        # 2. TTS for Narration
        tts_model = load_tts()
        audio_paths = []
        for i, scene in enumerate(narration_scenes):
            audio_file = os.path.join(config.output_dir, f"scene_{i}_audio.wav")
            generate_audio(
                scene["text"], 
                tts_model, 
                audio_file, 
                speaker_wav=speaker_reference_audio if "xtts" in TTS_MODEL_ID.lower() else None
            )
            audio_paths.append(audio_file)
        clear_vram(tts_model)

        # 3. Visual Generation
        video_clip_paths = []
        if config.use_svd_flow: # Image (SDXL) -> Video (SVD)
            t2i_pipe, refiner = load_t2i_pipeline() # SDXL
            i2v_pipe = load_i2v_pipeline()         # SVD
            for i, (visual_prompt, scene) in enumerate(zip(visual_prompts_scenes, narration_scenes)):
                clip_path = generate_image_then_video(
                    visual_prompt, 
                    i2v_pipe, 
                    t2i_pipe, 
                    refiner, 
                    i,
                    scene["duration"],
                    config
                )
                video_clip_paths.append(clip_path)
            clear_vram(t2i_pipe, refiner, i2v_pipe)
        else: # Direct Text-to-Video (ModelScope)
            t2v_pipe = load_t2v_pipeline()
            for i, (visual_prompt, scene) in enumerate(zip(visual_prompts_scenes, narration_scenes)):
                clip_path = generate_direct_video(
                    visual_prompt, 
                    t2v_pipe, 
                    i,
                    scene["duration"],
                    config
                )
                video_clip_paths.append(clip_path)
            clear_vram(t2v_pipe)

        # 4. Video Assembly
        if video_clip_paths and audio_paths:
            final_video_path = assemble_final_video(
                video_clip_paths, 
                audio_paths, 
                narration_scenes,
                config
            )
        else:
            print("Not enough assets generated to assemble video.")

        # 5. Output final info
        if final_video_path:
            print("\n--- AUTOMATION COMPLETE ---")
            print(f"Final Video: {final_video_path}")
            print(f"Suggested Instagram Caption Text:\n{' '.join([scene['text'] for scene in narration_scenes])}")
            print(f"Suggested Hashtags: {', '.join(hashtags)}")
        else:
            print("\n--- AUTOMATION FAILED ---")
            print("Check logs for errors.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure all models are cleared from VRAM if they were loaded
        print("Cleaning up any remaining models from VRAM...")
        models_to_clear = [m for m in [llm, tts_model, t2i_pipe, refiner, i2v_pipe, t2v_pipe] if m is not None]
        if models_to_clear:
            clear_vram(*models_to_clear)
        print("Cleanup finished.")



if __name__ == "__main__":
    # --- IMPORTANT SETUP ---
    # 1. Install dependencies:
    #    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    #    pip install transformers accelerate bitsandbytes diffusers TTS moviepy Pillow safetensors sentencepiece
    #    (Make sure CUDA version in pytorch matches your system's CUDA Toolkit)
    # 2. Download models:
    #    The first time you run, Hugging Face models will download.
    #    For Coqui TTS, ensure you have the model files (XTTSv2 needs manual download or will try).
    # 3. For XTTSv2 voice cloning:
    #    Set `speaker_reference_audio` to a path of a clean WAV file of the voice you want to clone.
    #    It must be a 16-bit PCM WAV file, ideally >15 seconds long.
    # 4. Fonts for MoviePy: Make sure 'Arial-Bold' or your chosen font is available to MoviePy/ImageMagick.
    #    If not, ImageMagick might need to be installed and configured, or use a default font.

    # --- RUN THE SCRIPT ---
    # Example configuration
    config = ContentConfig(
        target_video_length=30.0,  # 30 seconds total
        max_scene_length=3.0,      # 3 seconds per scene
        min_scenes=2,
        max_scenes=3,
        use_svd_flow=True,         # Use SDXL -> SVD flow
        fps=24
    )

    topic_for_reel = "benefits of meditation"
    main_automation_flow(topic_for_reel, config)
