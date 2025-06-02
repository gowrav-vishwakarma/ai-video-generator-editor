#!/usr/bin/env python
# coding: utf-8

# main_video_generator.py
import os
import importlib
import math
import time
from typing import Optional, List, Dict, Any

# Assume video_assembly.py is in the same directory or accessible in PYTHONPATH
from video_assembly import assemble_final_reel, assemble_scene_video_from_sub_clips
from config_manager import ContentConfig, ModuleSelectorConfig, DEVICE, clear_vram_globally
from moviepy import VideoFileClip # For getting duration

# --- Utility to dynamically load modules ---
def load_module(module_path_str: str):
    try:
        module = importlib.import_module(module_path_str)
        print(f"Successfully loaded module: {module_path_str}")
        return module
    except ImportError as e:
        print(f"Error loading module {module_path_str}: {e}")
        raise

# --- MAIN WORKFLOW ---
def main_automation_flow(
    topic: str, 
    content_cfg: ContentConfig, 
    module_selector_cfg: ModuleSelectorConfig,
    speaker_reference_audio: Optional[str] = None
):
    final_video_path = None
    
    # Dynamically load selected modules
    llm_module = load_module(module_selector_cfg.llm_module)
    tts_module = load_module(module_selector_cfg.tts_module)
    t2i_module = load_module(module_selector_cfg.t2i_module)
    i2v_module = load_module(module_selector_cfg.i2v_module)
    t2v_module = load_module(module_selector_cfg.t2v_module)

    # Get configs from the loaded modules
    llm_cfg = llm_module.LLMConfig()
    tts_cfg = tts_module.TTSConfig()
    t2i_cfg = t2i_module.T2IConfig()
    i2v_cfg = i2v_module.I2VConfig()
    t2v_cfg = t2v_module.T2VConfig()

    # Create output directory from content_cfg
    os.makedirs(content_cfg.output_dir, exist_ok=True)

    try:
        # 1. LLM for Script and Prompts
        script_narration_parts, script_visual_prompts, hashtags = llm_module.generate_script(
            topic, content_cfg, llm_cfg
        )
        llm_module.clear_llm_vram() # Clear LLM after initial script generation
        
        processed_scene_assets = []

        # 2. TTS for Narration (Load once, use for all scenes)
        # TTS model will be loaded on first call to generate_audio within its module

        # --- Loop through each scripted scene ---
        for i, (narration_info, visual_prompt_for_scene) in enumerate(zip(script_narration_parts, script_visual_prompts)):
            scene_start_time = time.time()
            print(f"\n--- Processing Scene {i+1}/{len(script_narration_parts)} ---")
            narration_text = narration_info["text"]
            
            # 2.a. Generate Audio & Get Actual Duration
            scene_audio_path, actual_audio_duration = tts_module.generate_audio(
                narration_text, content_cfg.output_dir, i, tts_cfg,
                speaker_wav=speaker_reference_audio
            )

            if actual_audio_duration <= 0.1:
                print(f"Scene {i+1} has negligible audio. Skipping video generation for this scene.")
                continue

            num_video_chunks = math.ceil(actual_audio_duration / content_cfg.model_max_video_chunk_duration)
            if num_video_chunks == 0: num_video_chunks = 1
            print(f"Scene {i+1}: Audio {actual_audio_duration:.2f}s. Needs {num_video_chunks} video chunk(s).")

            # 2.b. Generate Chunk-Specific Visual Prompts using LLM
            # LLM model will be re-loaded by its module if cleared
            chunk_specific_prompts = llm_module.generate_chunk_visual_prompts(
                narration_text, visual_prompt_for_scene, num_video_chunks, content_cfg, llm_cfg
            )
            llm_module.clear_llm_vram() # Clear LLM after generating chunk prompts

            video_sub_clip_paths_for_scene = []
            current_scene_audio_covered_duration = 0.0
            
            gen_width, gen_height = content_cfg.generation_resolution

            for chunk_idx in range(num_video_chunks):
                print(f"  Generating video chunk {chunk_idx+1}/{num_video_chunks} for scene {i+1}...")
                
                if chunk_idx < num_video_chunks - 1:
                    current_chunk_target_duration = content_cfg.model_max_video_chunk_duration
                else:
                    current_chunk_target_duration = actual_audio_duration - current_scene_audio_covered_duration
                current_chunk_target_duration = max(0.5, current_chunk_target_duration)

                visual_prompt, motion_prompt = chunk_specific_prompts[chunk_idx]  # Unpack both prompts
                sub_clip_path = None
                num_frames_for_chunk = max(i2v_cfg.svd_min_frames if content_cfg.use_svd_flow else 8, 
                                           int(current_chunk_target_duration * content_cfg.fps))


                if content_cfg.use_svd_flow:
                    # Generate Keyframe Image
                    keyframe_image_filename = f"scene_{i}_chunk_{chunk_idx}_keyframe.png"
                    keyframe_image_path = os.path.join(content_cfg.output_dir, keyframe_image_filename)
                    
                    t2i_module.generate_image(
                        visual_prompt, keyframe_image_path, 
                        gen_width, gen_height, t2i_cfg
                    )
                    # T2I model loaded/cleared within its module per call if not managed globally

                    # Generate Video from Image
                    video_chunk_filename = f"scene_{i}_chunk_{chunk_idx}_svd.mp4"
                    video_chunk_path = os.path.join(content_cfg.output_dir, video_chunk_filename)
                    
                    sub_clip_path = i2v_module.generate_video_from_image(
                        keyframe_image_path, video_chunk_path,
                        num_frames_for_chunk, content_cfg.fps,
                        gen_width, gen_height, i2v_cfg,
                        motion_prompt=motion_prompt  # Pass the motion prompt
                    )
                    # I2V model loaded/cleared within its module per call
                else: # Direct T2V
                    video_chunk_filename = f"scene_{i}_chunk_{chunk_idx}_t2v.mp4"
                    video_chunk_path = os.path.join(content_cfg.output_dir, video_chunk_filename)
                    sub_clip_path = t2v_module.generate_video_from_text(
                        visual_prompt, video_chunk_path,
                        num_frames_for_chunk, content_cfg.fps,
                        gen_width, gen_height, t2v_cfg
                    )
                    # T2V model loaded/cleared within its module per call

                if sub_clip_path and os.path.exists(sub_clip_path):
                    video_sub_clip_paths_for_scene.append(sub_clip_path)
                    with VideoFileClip(sub_clip_path) as temp_vfc:
                        generated_chunk_duration = temp_vfc.duration
                    current_scene_audio_covered_duration += generated_chunk_duration
                else:
                    print(f"    Failed to generate video chunk {chunk_idx+1} for scene {i+1}.")
            
            # Clear visual models after processing all chunks for a scene
            if content_cfg.use_svd_flow:
                t2i_module.clear_t2i_vram()
                i2v_module.clear_i2v_vram()
            else:
                t2v_module.clear_t2v_vram()

            if video_sub_clip_paths_for_scene:
                final_video_for_scene_path = assemble_scene_video_from_sub_clips(
                    video_sub_clip_paths_for_scene, actual_audio_duration, content_cfg, i
                ) # content_cfg now passed to assembly
                if final_video_for_scene_path:
                    narration_data_for_assembly = {'text': narration_text, 'duration': actual_audio_duration}
                    processed_scene_assets.append((final_video_for_scene_path, scene_audio_path, narration_data_for_assembly))
                else:
                    print(f"Failed to assemble video for scene {i+1}.")
            else:
                print(f"No video sub-clips for scene {i+1}. Skipping assembly.")
            
            print(f"--- Scene {i+1} processing took {time.time() - scene_start_time:.2f}s ---")

        # Clear TTS model after all scenes are processed
        tts_module.clear_tts_vram()

        if processed_scene_assets:
            final_video_path = assemble_final_reel(
                processed_scene_assets, content_cfg, # Pass content_cfg
                output_filename=f"{topic.replace(' ','_').replace('.', '')}_final_reel.mp4"
            )
        else:
            print("No scenes processed. Cannot create final video.")

        if final_video_path:
            print("\n--- AUTOMATION COMPLETE ---")
            print(f"Final Video: {final_video_path}")
            full_narration_text = " ".join([asset[2]["text"] for asset in processed_scene_assets])
            print(f"Suggested Caption Text:\n{full_narration_text}")
            print(f"Suggested Hashtags: {', '.join(hashtags)}")
        else:
            print("\n--- AUTOMATION FAILED OR NO OUTPUT ---")

    except Exception as e:
        print(f"Unhandled error in main_automation_flow: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Final cleanup: Ensuring all modules attempt VRAM clearing...")
        # Call clear methods on all loaded module objects, they handle their own state
        if 'llm_module' in locals() and hasattr(llm_module, 'clear_llm_vram'): llm_module.clear_llm_vram()
        if 'tts_module' in locals() and hasattr(tts_module, 'clear_tts_vram'): tts_module.clear_tts_vram()
        if 't2i_module' in locals() and hasattr(t2i_module, 'clear_t2i_vram'): t2i_module.clear_t2i_vram()
        if 'i2v_module' in locals() and hasattr(i2v_module, 'clear_i2v_vram'): i2v_module.clear_i2v_vram()
        if 't2v_module' in locals() and hasattr(t2v_module, 'clear_t2v_vram'): t2v_module.clear_t2v_vram()
        
        # A final global clear just in case, though module clears should handle it
        clear_vram_globally() 
        print("Cleanup finished.")
    
    return final_video_path


if __name__ == "__main__":
    print(f"Running on device: {DEVICE}")

    # --- Centralized Configuration ---
    content_settings = ContentConfig(
        target_video_length_hint=15,         
        model_max_video_chunk_duration=2.5,   # SVD-XT typically good for 25 frames (2.5s at 10fps)
        max_scene_narration_duration_hint=7.0,
        min_scenes=2,
        max_scenes=4, # Keep it small for testing
        use_svd_flow=True, # True: SDXL -> SVD; False: Zeroscope T2V
        fps=10, # SVD is often trained at lower FPS like 7, then upsample or export at higher
        # generation_resolution=(1024, 576), # Optimal for SDXL and SVD-XT (16:9)
        # final_output_resolution=(1280, 720), # e.g., 720p for faster final assembly
        generation_resolution=(576, 1024), # Optimal for SDXL and SVD-XT (9:16 for reels)
        final_output_resolution=(1080, 1920), # Instagram reels standard resolution
        output_dir="modular_reels_output",
        font_for_subtitles="Arial" # Make sure this font is available or provide path to .ttf
    )

    module_choices = ModuleSelectorConfig() # Uses default modules and their configs

    # --- Customize module specific configs if needed ---
    # Example: Use a different LLM model or settings
    # module_choices.llm_config.model_id = "mistralai/Mistral-7B-Instruct-v0.2" 
    # module_choices.llm_config.max_new_tokens_script = 2048

    # Example: T2I settings (if using SDXL)
    # module_choices.t2i_config.num_inference_steps = 25
    # module_choices.t2i_config.refiner_id = "stabilityai/stable-diffusion-xl-refiner-1.0" # Enable refiner

    # Example: SVD settings (if use_svd_flow is True)
    # module_choices.i2v_config.motion_bucket_id = 180 # Higher motion
    # module_choices.i2v_config.svd_max_frames = 14 # If using base SVD not SVD-XT

    # Example: T2V settings (if use_svd_flow is False)
    # module_choices.t2v_config.model_id = "damo-vilab/text-to-video-ms-1.7b"
    # if module_choices.t2v_config.model_id == "damo-vilab/text-to-video-ms-1.7b":
    #    content_settings.generation_resolution = (256,256) # ModelScope T2V is often 256x256

    speaker_audio_sample = "record_out.wav" # Path to your speaker reference WAV
    if not os.path.exists(speaker_audio_sample):
        print(f"Warning: Speaker ref audio '{speaker_audio_sample}' not found. XTTS uses default voice.")
        speaker_audio_sample = None
    
    reel_topic = "it can be funny if you are a cat, human thinking like cats."
    # reel_topic = "explaining quantum entanglement simply"
    
    start_time = time.time()
    generated_video = main_automation_flow(
        reel_topic, 
        content_settings, 
        module_choices,
        speaker_reference_audio=speaker_audio_sample
    )
    end_time = time.time()

    if generated_video:
        print(f"Successfully generated video: {generated_video}")
    else:
        print("Video generation failed or produced no output.")
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")