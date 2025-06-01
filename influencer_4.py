#!/usr/bin/env python
# coding: utf-8

# Developed by Gowrav Vishwakarma 
# https://www.linkedin.com/in/gowravvishwakarma/

# main_script.py
import os

from video_assembly import assemble_final_reel, assemble_scene_video_from_sub_clips
from config import ContentConfig

# Set PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import math # For math.ceil
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionXLPipeline, StableVideoDiffusionPipeline, DiffusionPipeline
from diffusers.utils import load_image, export_to_video
from TTS.api import TTS # Coqui TTS
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from moviepy.audio.AudioClip import concatenate_audioclips, AudioClip 
from moviepy.video.fx.Crop import Crop # Correct import based on your docs
import gc # Garbage collection
import numpy as np
import time # For unique filenames if needed, and timing

# --- 0. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# LLM paths
LLM_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
# TTS model
TTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
# T2I model
T2I_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
T2I_REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0" # Optional
# I2V model (SVD)
I2V_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
# T2V model (alternative)
# T2V_MODEL_ID = "damo-vilab/text-to-video-ms-1.7b" # Example if using ModelScope
T2V_MODEL_ID = "cerspense/zeroscope_v2_576w" # Another option, or use original THUDM/CogVideoX-5b if available/preferred

# --- 1. INITIALIZE MODELS (Load only when needed or keep loaded if VRAM allows) ---
def load_llm():
    print("Loading LLM...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    return model, tokenizer

def load_tts():
    print("Loading TTS model...")
    return TTS(model_name=TTS_MODEL_ID, progress_bar=True).to(DEVICE)

def load_t2i_pipeline():
    print("Loading T2I pipeline (SDXL)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        T2I_MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(DEVICE)
    # Optional: Load refiner
    # refiner = DiffusionPipeline.from_pretrained(
    #     T2I_REFINER_ID, text_encoder_2=pipe.text_encoder_2, vae=pipe.vae,
    #     torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    # ).to(DEVICE)
    # return pipe, refiner
    return pipe, None 

def load_i2v_pipeline(): # SVD
    print("Loading I2V pipeline (SVD)...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        I2V_MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    ).to(DEVICE)
    pipe.enable_model_cpu_offload()
    return pipe

def load_t2v_pipeline():
    print(f"Loading T2V pipeline ({T2V_MODEL_ID})...")
    pipe = DiffusionPipeline.from_pretrained(
        T2V_MODEL_ID, torch_dtype=torch.float16 
        # variant="fp16" # if applicable
    ).to(DEVICE)
    pipe.enable_model_cpu_offload()
    return pipe

# --- UTILITY TO CLEAR VRAM ---
def clear_vram(*models_or_pipelines):
    for item in models_or_pipelines:
        if item is None:
            continue
        try:
            if hasattr(item, 'cpu') and callable(getattr(item, 'cpu')):
                item.cpu()
            elif hasattr(item, 'model') and hasattr(item.model, 'cpu') and callable(getattr(item.model, 'cpu')):
                item.model.cpu()
        except Exception as e:
            print(f"Warning: Could not move model to CPU: {e}")
            # Try to delete the model directly if moving to CPU fails
            try:
                del item
            except:
                pass
    # Aggressively clear
    del models_or_pipelines
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("VRAM cleared and memory collected.")


# --- 2. TEXT GENERATION (SCRIPTING) ---
def generate_script_from_llm(topic: str, model, tokenizer, config: ContentConfig) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    print(f"Generating script and prompts for topic: {topic}")
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant creating content for a short video. "
                "Your response must be in valid JSON format: "
                "{\"narration\": [{\"scene\": 1, \"text\": \"narration_text\", \"duration_estimate\": float_seconds}], "
                "\"visuals\": [{\"scene\": 1, \"prompt\": \"visual_prompt_for_scene\"}], "
                "\"hashtags\": [\"tag1\", \"tag2\"]}"
            )
        },
        {
            "role": "user",
            "content": f"""
            Create content for a short video about "{topic}".
            The video should be engaging and the total narration should be around {config.target_video_length_hint} seconds long.
            Each narration segment should ideally be around {config.max_scene_narration_duration_hint} seconds.
            Generate between {config.min_scenes} and {config.max_scenes} narration/visual scenes.
            
            Return your response in this exact JSON format:
            {{
                "narration": [
                    {{"scene": 1, "text": "First scene narration text.", "duration_estimate": {config.max_scene_narration_duration_hint}}},
                    {{"scene": 2, "text": "Second scene narration text.", "duration_estimate": {config.max_scene_narration_duration_hint}}}
                ],
                "visuals": [
                    {{"scene": 1, "prompt": "Detailed visual prompt for scene 1, focusing on key elements, style (e.g., cinematic, anime, realistic), camera angles, and mood."}},
                    {{"scene": 2, "prompt": "Detailed visual prompt for scene 2, similarly descriptive."}}
                ],
                "hashtags": ["relevantTag1", "relevantTag2", "relevantTag3"]
            }}
            
            Ensure the 'duration_estimate' is a float representing seconds.
            Each visual prompt should be detailed and suitable for image/video generation models.
            The number of narration entries must match the number of visual entries.
            """
        }
    ]

    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "input_ids": tokenized_chat, "max_new_tokens": 1536, "do_sample": True,
        "top_k": 50, "top_p": 0.95, "temperature": 0.7, "pad_token_id": tokenizer.eos_token_id
    }

    outputs = model.generate(**generation_kwargs)
    decoded_output = tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)
    print("LLM Response:\n", decoded_output)
    
    import re
    match = re.search(r'\{[\s\S]*\}', decoded_output) # More robust regex for finding first { to last }
    
    if match:
        json_text_to_parse = match.group(0)
        print(f"Extracted JSON block for parsing:\n{json_text_to_parse}")
    else:
        json_text_to_parse = decoded_output # Fallback if no clear block
        print(f"Could not extract a clear JSON block, attempting to parse full output.")

    try:
        response_data = json.loads(json_text_to_parse)
        narration_data = sorted(response_data.get("narration", []), key=lambda x: x["scene"])
        visuals_data = sorted(response_data.get("visuals", []), key=lambda x: x["scene"])
        
        narration_scenes = [{"text": s["text"], "duration_estimate": float(s.get("duration_estimate", config.max_scene_narration_duration_hint))} for s in narration_data]
        visual_prompts = [s["prompt"] for s in visuals_data]
        hashtags = response_data.get("hashtags", [])

        if len(narration_scenes) != len(visual_prompts):
            raise ValueError("Mismatch between number of narration and visual scenes.")
        if not (config.min_scenes <= len(narration_scenes) <= config.max_scenes):
            raise ValueError(f"Scene count {len(narration_scenes)} outside range [{config.min_scenes}, {config.max_scenes}].")
            
        return narration_scenes, visual_prompts, hashtags
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Error parsing LLM response: {e}. Using fallback content.")
        fallback_duration = config.max_scene_narration_duration_hint
        narration_scenes = [
            {"text": f"An introduction to {topic}.", "duration_estimate": fallback_duration},
            {"text": "Exploring details and implications.", "duration_estimate": fallback_duration}
        ]
        visual_prompts = [
            f"Cinematic overview of {topic}, vibrant colors, high detail.",
            f"Close-up abstract representation related to {topic}, thought-provoking."
        ]
        hashtags = [f"#{topic.replace(' ', '')}", "#AIvideo", "#GeneratedContent"]
        return narration_scenes, visual_prompts, hashtags


# --- 3. AUDIO GENERATION ---
def generate_audio_for_scene(text: str, tts_model, output_dir: str, scene_idx: int, speaker_wav: Optional[str] = None) -> Tuple[str, float]:
    print(f"Generating audio for scene {scene_idx}: \"{text[:50]}...\"")
    output_path = os.path.join(output_dir, f"scene_{scene_idx}_audio.wav")
    
    if "xtts" in TTS_MODEL_ID.lower():
        if speaker_wav and os.path.exists(speaker_wav):
            tts_model.tts_to_file(text, speaker_wav=speaker_wav, language="en", file_path=output_path)
        else:
            if speaker_wav: print(f"Warning: Speaker WAV {speaker_wav} not found for XTTS. Using default voice.")
            else: print("Warning: XTTS model selected but no speaker_wav provided. Using default voice.")
            tts_model.tts_to_file(text, language="en", file_path=output_path) # Default voice
    else:
        tts_model.tts_to_file(text, file_path=output_path)
    
    print(f"Audio for scene {scene_idx} saved to {output_path}")
    
    # Get actual audio duration
    try:
        with AudioFileClip(output_path) as audio_clip:
            duration = audio_clip.duration
        # A small buffer, Coqui TTS sometimes clips the very end.
        # This is optional, can be removed if not an issue.
        duration += 0.1 
    except Exception as e:
        print(f"Error getting duration for {output_path}: {e}. Assuming 0 duration.")
        duration = 0.0
        # Fallback: if file is corrupt or empty, it might cause issues later.
        # Consider creating a tiny valid silent wav if duration is 0 to prevent crashes.
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
             # Create a minimal valid WAV file if generation failed catastrophically
            from scipy.io import wavfile
            samplerate = 22050 # A common samplerate
            silence_data = np.zeros(int(0.1 * samplerate), dtype=np.int16) # 0.1s silence
            wavfile.write(output_path, samplerate, silence_data)
            print(f"Wrote fallback silent audio to {output_path}")
            duration = 0.1


    print(f"Actual audio duration for scene {scene_idx}: {duration:.2f}s")
    return output_path, duration


# --- 4. VISUAL GENERATION (PER CHUNK) ---
def generate_image_then_video_chunk(
    image_prompt: str, i2v_pipe, t2i_pipe, refiner_pipe,
    scene_idx: int, chunk_idx: int, target_chunk_duration: float, config: ContentConfig
) -> str:
    # Generate Image (Keyframe)
    print(f"Scene {scene_idx}, Chunk {chunk_idx}: Generating keyframe image for: \"{image_prompt[:50]}...\"")
    keyframe_image_path = os.path.join(config.output_dir, f"scene_{scene_idx}_chunk_{chunk_idx}_keyframe.png")

    # Calculate base dimensions (half of target resolution)
    base_width = config.target_resolution[0] // 2
    base_height = config.target_resolution[1] // 2

    # Ensure dimensions are divisible by 8 for SDXL
    # We can round down to the nearest multiple of 8
    sdxl_width = (base_width // 8) * 8
    sdxl_height = (base_height // 8) * 8

    # Make sure dimensions are not too small (e.g., SDXL might have min requirements like 256 or 512)
    # This is a sensible default, adjust if your model needs larger minimums
    sdxl_width = max(sdxl_width, 512 if sdxl_width > sdxl_height else 256) # Example min logic
    sdxl_height = max(sdxl_height, 512 if sdxl_height > sdxl_width else 256) # Example min logic
    # Ensure they are still multiples of 8 after max()
    sdxl_width = (sdxl_width // 8) * 8
    sdxl_height = (sdxl_height // 8) * 8


    print(f"Scene {scene_idx}, Chunk {chunk_idx}: SDXL target image size: {sdxl_width}x{sdxl_height}")

    image = t2i_pipe(
        prompt=image_prompt, num_inference_steps=30, guidance_scale=7.5,
        width=sdxl_width, height=sdxl_height
    ).images[0]

    # if refiner_pipe: image = refiner_pipe(prompt=image_prompt, image=image[None, :]).images[0]
    image.save(keyframe_image_path)
    print(f"Keyframe image for scene {scene_idx}, chunk {chunk_idx} saved to {keyframe_image_path}")

    # Prepare image for SVD (load and resize if needed)
    loaded_image = load_image(keyframe_image_path)
    # SVD has its own preferred resolutions, often 1024x576 or 576x1024.
    # The image generated by SDXL will be loaded and SVD pipeline will handle resizing internally
    # if needed, or you can explicitly resize it here if you know SVD's exact optimal input.
    # For SVD-XT, common input sizes are around 1024x576 or 576x1024.
    # If your sdxl_width/height are far from this, SVD might resize, potentially affecting quality.
    # Example explicit resize for SVD if needed:
    # if sdxl_width > sdxl_height: # Landscape-ish
    #     svd_input_image = loaded_image.resized((1024, 576)) # Example
    # else: # Portrait-ish
    #     svd_input_image = loaded_image.resized((576, 1024)) # Example
    # But usually, just passing `loaded_image` is fine.
    svd_input_image = loaded_image


    num_frames = max(8, int(target_chunk_duration * config.fps))
    num_frames = min(num_frames, 25) # SVD-XT typically max 25 frames

    print(f"Scene {scene_idx}, Chunk {chunk_idx}: Generating video from image ({num_frames} frames) using SVD...")
    video_frames = i2v_pipe(
        svd_input_image, decode_chunk_size=4, num_frames=num_frames, motion_bucket_id=127,
        fps=config.fps, noise_aug_strength=0.02
    ).frames[0]

    video_chunk_path = os.path.join(config.output_dir, f"scene_{scene_idx}_chunk_{chunk_idx}_svd.mp4")
    export_to_video(video_frames, video_chunk_path, fps=config.fps)
    print(f"SVD video chunk for scene {scene_idx}, chunk {chunk_idx} saved to {video_chunk_path}")
    return video_chunk_path

def generate_direct_video_chunk(
    video_prompt: str, t2v_pipe, 
    scene_idx: int, chunk_idx: int, target_chunk_duration: float, config: ContentConfig
) -> str:
    print(f"Scene {scene_idx}, Chunk {chunk_idx}: Generating direct video for: \"{video_prompt[:50]}...\"")
    
    num_frames = max(8, int(target_chunk_duration * config.fps)) # Min frames
    # Some T2V models also have max frame limits or work best in certain ranges
    # num_frames = min(num_frames, 60) # Example: if model supports up to 60 frames

    # ModelScope/Zeroscope might need width/height parameters
    video_frames = t2v_pipe(
        prompt=video_prompt, 
        num_inference_steps=25, 
        num_frames=num_frames,
        height=config.target_resolution[1] // 2, # Example: common T2V models often prefer smaller resolutions
        width=config.target_resolution[0] // 2
    ).frames # For many diffusers T2V, .frames is the list of PILs

    video_chunk_path = os.path.join(config.output_dir, f"scene_{scene_idx}_chunk_{chunk_idx}_t2v.mp4")
    export_to_video(video_frames, video_chunk_path, fps=config.fps)
    print(f"T2V video chunk for scene {scene_idx}, chunk {chunk_idx} saved to {video_chunk_path}")
    return video_chunk_path

def generate_chunk_specific_visual_prompts(
    scene_narration: str,
    num_chunks: int,
    model,
    tokenizer,
    config: ContentConfig
) -> List[str]:
    """
    Generate specific visual prompts for each chunk of a scene based on its position in the narration.
    """
    print(f"Generating {num_chunks} chunk-specific visual prompts for scene narration...")
    
    chunk_prompts = []
    
    for chunk_idx in range(num_chunks):
        print(f"\nGenerating prompt for chunk {chunk_idx + 1}/{num_chunks}...")
        
        # Calculate approximate timing for this chunk
        chunk_start = chunk_idx * config.model_max_video_chunk_duration
        chunk_end = min((chunk_idx + 1) * config.model_max_video_chunk_duration, 
                       len(scene_narration.split()) * 0.3)  # Rough estimate: 0.3s per word
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant creating a visual prompt for a specific chunk of a video. "
                    "Your task is to create a detailed visual prompt that captures the essence of the narration "
                    "for this specific time segment."
                )
            },
            {
                "role": "user",
                "content": f"""
                Given this narration: "{scene_narration}"
                
                Create a detailed visual prompt for chunk {chunk_idx + 1} of {num_chunks}.
                This chunk represents approximately {chunk_start:.1f}s to {chunk_end:.1f}s of the narration.
                
                Return your response in this exact JSON format:
                {{
                    "prompt": "Detailed visual prompt focusing on the specific part of the narration for this chunk"
                }}
                
                The prompt should be detailed and suitable for image/video generation models.
                Focus on the specific part of the narration that would be shown during this chunk's timing.
                """
            }
        ]

        try:
            tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
            generation_kwargs = {
                "input_ids": tokenized_chat, "max_new_tokens": 512, "do_sample": True,
                "top_k": 50, "top_p": 0.95, "temperature": 0.7, "pad_token_id": tokenizer.eos_token_id
            }

            outputs = model.generate(**generation_kwargs)
            decoded_output = tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)
            
            # Debug: Print raw LLM output
            print(f"\nRaw LLM output for chunk {chunk_idx + 1}:")
            print("-" * 50)
            print(decoded_output)
            print("-" * 50)
            
            import re
            match = re.search(r'\{[\s\S]*\}', decoded_output)
            
            if match:
                json_text_to_parse = match.group(0)
            else:
                json_text_to_parse = decoded_output

            try:
                response_data = json.loads(json_text_to_parse)
                prompt = response_data.get("prompt")
                if prompt:
                    chunk_prompts.append(prompt)
                    print(f"Successfully generated prompt for chunk {chunk_idx + 1}")
                else:
                    raise ValueError("No prompt found in response")
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Error parsing LLM response for chunk {chunk_idx + 1}: {e}")
                # Create a fallback prompt for this specific chunk
                if chunk_idx == 0:
                    prompt = f"Visual representation of the beginning of the scene: {scene_narration[:50]}..."
                elif chunk_idx == num_chunks - 1:
                    prompt = f"Visual representation of the conclusion of the scene: {scene_narration[-50:]}..."
                else:
                    prompt = f"Visual representation of part {chunk_idx + 1} of {num_chunks} of the scene: {scene_narration[:50]}..."
                chunk_prompts.append(prompt)
                print(f"Using fallback prompt for chunk {chunk_idx + 1}")
                
        except Exception as e:
            print(f"Error generating prompt for chunk {chunk_idx + 1}: {e}")
            # Create a fallback prompt for this specific chunk
            if chunk_idx == 0:
                prompt = f"Visual representation of the beginning of the scene: {scene_narration[:50]}..."
            elif chunk_idx == num_chunks - 1:
                prompt = f"Visual representation of the conclusion of the scene: {scene_narration[-50:]}..."
            else:
                prompt = f"Visual representation of part {chunk_idx + 1} of {num_chunks} of the scene: {scene_narration[:50]}..."
            chunk_prompts.append(prompt)
            print(f"Using fallback prompt for chunk {chunk_idx + 1}")

    print("\nFinal chunk prompts:")
    for i, prompt in enumerate(chunk_prompts):
        print(f"\nChunk {i + 1}:")
        print(prompt)
    
    return chunk_prompts

# --- MAIN WORKFLOW ---
def main_automation_flow(topic: str, config: ContentConfig, speaker_reference_audio: Optional[str] = None):
    final_video_path = None
    # Initialize model placeholders
    llm_model, llm_tokenizer = None, None
    tts_model = None
    t2i_pipe, refiner = None, None
    i2v_pipe = None
    t2v_pipe = None

    try:
        # 1. LLM for Script and Prompts
        llm_model, llm_tokenizer = load_llm()
        script_narration_parts, script_visual_prompts, hashtags = generate_script_from_llm(
            topic, llm_model, llm_tokenizer, config
        )
        # Clear LLM after initial script generation
        clear_vram(llm_model, llm_tokenizer)
        llm_model, llm_tokenizer = None, None
        
        # Prepare for collecting assets for each scene
        processed_scene_assets = [] # List of (final_scene_video_path, scene_audio_path, narration_info_dict)

        # 2. TTS for Narration (Load once, use for all scenes)
        tts_model = load_tts()
        
        # Load video generation models
        if config.use_svd_flow:
            t2i_pipe, refiner = load_t2i_pipeline()
            i2v_pipe = load_i2v_pipeline()
        else:
            t2v_pipe = load_t2v_pipeline()

        # --- Loop through each scripted scene ---
        for i, (narration_info, visual_prompt_for_scene) in enumerate(zip(script_narration_parts, script_visual_prompts)):
            scene_start_time = time.time()
            print(f"\n--- Processing Scene {i+1}/{len(script_narration_parts)} ---")
            narration_text = narration_info["text"]
            
            # 2.a. Generate Audio for this narration part & Get Actual Duration
            scene_audio_path, actual_audio_duration = generate_audio_for_scene(
                narration_text, tts_model, config.output_dir, i,
                speaker_wav=speaker_reference_audio
            )

            if actual_audio_duration <= 0.1: # Threshold for too short/failed audio
                print(f"Scene {i} has negligible audio duration ({actual_audio_duration:.2f}s). Skipping video generation for this scene.")
                continue

            # 2.b. Determine number of video chunks needed for this audio duration
            num_video_chunks = math.ceil(actual_audio_duration / config.model_max_video_chunk_duration)
            if num_video_chunks == 0: num_video_chunks = 1 # Ensure at least one chunk if audio exists
            
            print(f"Scene {i}: Audio duration {actual_audio_duration:.2f}s. Needs {num_video_chunks} video chunk(s).")

            # Clear video models before loading LLM for chunk prompts
            if config.use_svd_flow:
                clear_vram(t2i_pipe, refiner, i2v_pipe)
                t2i_pipe, refiner, i2v_pipe = None, None, None
            else:
                clear_vram(t2v_pipe)
                t2v_pipe = None

            # Load LLM for chunk-specific prompts
            llm_model, llm_tokenizer = load_llm()
            chunk_specific_prompts = generate_chunk_specific_visual_prompts(
                narration_text, num_video_chunks, llm_model, llm_tokenizer, config
            )
            # Clear LLM after generating chunk prompts
            clear_vram(llm_model, llm_tokenizer)
            llm_model, llm_tokenizer = None, None

            # Reload video models for chunk generation
            if config.use_svd_flow:
                t2i_pipe, refiner = load_t2i_pipeline()
                i2v_pipe = load_i2v_pipeline()
            else:
                t2v_pipe = load_t2v_pipeline()

            video_sub_clip_paths_for_scene = []
            current_scene_audio_covered_duration = 0.0

            for chunk_idx in range(num_video_chunks):
                print(f"  Generating video chunk {chunk_idx+1}/{num_video_chunks} for scene {i}...")
                
                # Determine duration for this specific video chunk
                if chunk_idx < num_video_chunks - 1:
                    current_chunk_target_duration = config.model_max_video_chunk_duration
                else: # Last chunk takes the remainder
                    current_chunk_target_duration = actual_audio_duration - current_scene_audio_covered_duration
                
                current_chunk_target_duration = max(0.5, current_chunk_target_duration) # Ensure a minimum practical duration for video models

                # Use the chunk-specific prompt instead of the scene-wide prompt
                sub_clip_visual_prompt = chunk_specific_prompts[chunk_idx]
                
                sub_clip_path = None
                if config.use_svd_flow:
                    sub_clip_path = generate_image_then_video_chunk(
                        sub_clip_visual_prompt, i2v_pipe, t2i_pipe, refiner,
                        i, chunk_idx, current_chunk_target_duration, config
                    )
                else:
                    sub_clip_path = generate_direct_video_chunk(
                        sub_clip_visual_prompt, t2v_pipe,
                        i, chunk_idx, current_chunk_target_duration, config
                    )
                
                if sub_clip_path and os.path.exists(sub_clip_path):
                    video_sub_clip_paths_for_scene.append(sub_clip_path)
                    with VideoFileClip(sub_clip_path) as temp_vfc:
                        generated_chunk_duration = temp_vfc.duration
                    current_scene_audio_covered_duration += generated_chunk_duration
                else:
                    print(f"    Failed to generate video chunk {chunk_idx} for scene {i}.")
            
            # 2.c. Assemble sub-clips for the current scene's video
            if video_sub_clip_paths_for_scene:
                final_video_for_scene_path = assemble_scene_video_from_sub_clips(
                    video_sub_clip_paths_for_scene, actual_audio_duration, config, i
                )
                if final_video_for_scene_path:
                    # Store asset paths for final assembly
                    narration_data_for_assembly = {'text': narration_text, 'duration': actual_audio_duration}
                    processed_scene_assets.append((final_video_for_scene_path, scene_audio_path, narration_data_for_assembly))
                else:
                    print(f"Failed to assemble video for scene {i} from its sub-clips.")
            else:
                print(f"No video sub-clips generated for scene {i}. Skipping assembly for this scene.")
            
            print(f"--- Scene {i+1} processing took {time.time() - scene_start_time:.2f}s ---")

        # Clear remaining models
        clear_vram(tts_model)
        tts_model = None
        if config.use_svd_flow:
            clear_vram(t2i_pipe, refiner, i2v_pipe)
            t2i_pipe, refiner, i2v_pipe = None, None, None
        else:
            clear_vram(t2v_pipe)
            t2v_pipe = None

        # 3. Assemble Final Video from all processed scenes
        if processed_scene_assets:
            final_video_path = assemble_final_reel(
                processed_scene_assets, config, output_filename=f"{topic.replace(' ','_')}_final_reel.mp4"
            )
        else:
            print("No scenes were successfully processed. Cannot create final video.")

        # 4. Output final info
        if final_video_path:
            print("\n--- AUTOMATION COMPLETE ---")
            print(f"Final Video: {final_video_path}")
            # Construct full narration text for caption
            full_narration_text = " ".join([asset[2]["text"] for asset in processed_scene_assets])
            print(f"Suggested Instagram Caption Text:\n{full_narration_text}")
            print(f"Suggested Hashtags: {', '.join(hashtags)}")
        else:
            print("\n--- AUTOMATION FAILED OR COMPLETED WITH NO OUTPUT ---")
            print("Check logs for errors or empty scene processing.")

    except Exception as e:
        print(f"An unhandled error occurred in main_automation_flow: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Final cleanup: Ensuring all models are cleared from VRAM...")
        models_to_clear = [m for m in [llm_model, llm_tokenizer, tts_model, t2i_pipe, refiner, i2v_pipe, t2v_pipe] if m is not None]
        if models_to_clear:
            clear_vram(*models_to_clear)
        print("Cleanup finished.")
    
    return final_video_path


if __name__ == "__main__":
    # --- IMPORTANT SETUP ---
    # (Ensure dependencies, models, CUDA, speaker WAV, and fonts are set up as per original comments)
    
    # Example configuration
    content_creation_config = ContentConfig(
        target_video_length_hint=5,          # Aim for ~20s total narration (LLM guidance)
        model_max_video_chunk_duration=3.0,     # Video models generate chunks up to 3s
        max_scene_narration_duration_hint=6.0,  # LLM hint: each narration part ~6s
        min_scenes=1,
        max_scenes=2,
        use_svd_flow=True,                      # SDXL -> SVD
        fps=10,                                  # Higher FPS for smoother SVD
        output_dir="my_video_project"
    )

    # For XTTSv2 voice cloning (replace with your WAV file path)
    # Ensure it's a clean, 16-bit PCM WAV, preferably >10s.
    speaker_audio_sample = "record_out.wav" # Path to your speaker reference WAV
    if not os.path.exists(speaker_audio_sample):
        print(f"Warning: Speaker reference audio '{speaker_audio_sample}' not found. XTTS will use a default voice.")
        speaker_audio_sample = None


    # reel_topic = "The future of renewable energy sources"
    reel_topic = "A fat cat saw a dress with she can't wear due to its obesity. and then she does gym and in last wear the dress."
    # reel_topic = "An 80 years old person guiding young generation."
    
    start_time = time.time()
    generated_video = main_automation_flow(reel_topic, content_creation_config, speaker_reference_audio=speaker_audio_sample)
    end_time = time.time()

    if generated_video:
        print(f"Successfully generated video: {generated_video}")
    else:
        print("Video generation failed or produced no output.")
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")