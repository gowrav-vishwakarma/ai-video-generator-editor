#!/usr/bin/env python
# coding: utf-8

# main_video_generator.py
import os
import importlib
import math
import time
import logging
from typing import Optional, List, Dict, Any

# Assume video_assembly.py is in the same directory or accessible in PYTHONPATH
from video_assembly import assemble_final_reel, assemble_scene_video_from_sub_clips
from config_manager import ContentConfig, ModuleSelectorConfig, DEVICE, clear_vram_globally
from moviepy import VideoFileClip # For getting duration
from project_manager import ProjectManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility to dynamically load modules ---
def load_module(module_path_str: str):
    try:
        module = importlib.import_module(module_path_str)
        logger.info(f"Successfully loaded module: {module_path_str}")
        return module
    except ImportError as e:
        logger.error(f"Error loading module {module_path_str}: {e}")
        raise

# --- MAIN WORKFLOW ---
def main_automation_flow(
    topic: str, 
    content_cfg: ContentConfig, 
    module_selector_cfg: ModuleSelectorConfig,
    speaker_reference_audio: Optional[str] = None,
    resume: bool = False
):
    final_video_path = None
    
    # Initialize project manager
    project_manager = ProjectManager(content_cfg.output_dir)
    
    if resume:
        if not project_manager.load_project():
            logger.warning("No existing project found to resume.")
            return None
        logger.info("Resuming existing project...")
    else:
        project_manager.initialize_project(topic, content_cfg)
        logger.info("Initialized new project...")
    
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

    try:
        while True:
            task, task_data = project_manager.get_next_pending_task()
            if not task:
                break
                
            logger.info(f"\n--- Processing task: {task} ---")
            
            if task == "generate_script":
                # 1. LLM for Script and Prompts
                script_narration_parts, script_visual_prompts, hashtags = llm_module.generate_script(
                    topic, content_cfg, llm_cfg
                )
                llm_module.clear_llm_vram()
                
                # Update project state with script
                narration_parts = [{"text": part["text"], "status": "pending"} 
                                 for part in script_narration_parts]
                visual_prompts = [{"prompt": prompt, "status": "pending"} 
                                for prompt in script_visual_prompts]
                project_manager.update_script(narration_parts, visual_prompts, hashtags)
                logger.info("Script generated and updated in project state")
                
            elif task == "generate_audio":
                scene_idx = task_data["scene_idx"]
                narration_text = task_data["text"]
                
                # Generate audio for the scene
                scene_audio_path, actual_audio_duration = tts_module.generate_audio(
                    narration_text, content_cfg.output_dir, scene_idx, tts_cfg,
                    speaker_wav=speaker_reference_audio
                )
                
                if actual_audio_duration <= 0.1:
                    logger.warning(f"Scene {scene_idx+1} has negligible audio. Skipping.")
                    continue
                    
                # Update narration part status
                narration_parts = project_manager.state.script["narration_parts"]
                narration_parts[scene_idx].update({
                    "audio_path": scene_audio_path,
                    "duration": actual_audio_duration,
                    "status": "generated"
                })
                project_manager.update_script(narration_parts, 
                                           project_manager.state.script["visual_prompts"],
                                           project_manager.state.script["hashtags"])
                logger.info(f"Audio generated for scene {scene_idx+1}")
                
            elif task == "create_scene":
                scene_idx = task_data["scene_idx"]
                narration = project_manager.state.script["narration_parts"][scene_idx]
                visual_prompt = project_manager.state.script["visual_prompts"][scene_idx]
                
                # Calculate number of chunks needed
                num_video_chunks = math.ceil(narration["duration"] / content_cfg.model_max_video_chunk_duration)
                if num_video_chunks == 0: num_video_chunks = 1
                logger.info(f"Scene {scene_idx+1}: Audio {narration['duration']:.2f}s. Needs {num_video_chunks} video chunk(s).")
                
                # Generate chunk-specific prompts
                chunk_specific_prompts = llm_module.generate_chunk_visual_prompts(
                    narration["text"], visual_prompt["prompt"], num_video_chunks, content_cfg, llm_cfg
                )
                llm_module.clear_llm_vram()
                
                # Create chunks
                chunks = []
                for chunk_idx in range(num_video_chunks):
                    if chunk_idx < num_video_chunks - 1:
                        current_chunk_target_duration = content_cfg.model_max_video_chunk_duration
                    else:
                        current_chunk_target_duration = narration["duration"] - (chunk_idx * content_cfg.model_max_video_chunk_duration)
                    current_chunk_target_duration = max(0.5, current_chunk_target_duration)
                    
                    visual_prompt, motion_prompt = chunk_specific_prompts[chunk_idx]
                    chunks.append({
                        "chunk_idx": chunk_idx,
                        "target_duration": current_chunk_target_duration,
                        "visual_prompt": visual_prompt,
                        "motion_prompt": motion_prompt,
                        "keyframe_image_path": "",
                        "video_path": "",
                        "status": "pending"
                    })
                
                # Update visual prompt status
                visual_prompts = project_manager.state.script["visual_prompts"]
                visual_prompts[scene_idx]["status"] = "generated"
                project_manager.update_script(
                    project_manager.state.script["narration_parts"],
                    visual_prompts,
                    project_manager.state.script["hashtags"]
                )
                
                # Add scene to project
                project_manager.add_scene(scene_idx, narration, chunks)
                logger.info(f"Created scene {scene_idx+1} with {num_video_chunks} chunks")
                
            elif task == "generate_chunk":
                scene_idx = task_data["scene_idx"]
                chunk_idx = task_data["chunk_idx"]
                visual_prompt = task_data["visual_prompt"]
                motion_prompt = task_data.get("motion_prompt")
                
                scene = project_manager.get_scene_info(scene_idx)
                chunk = project_manager.get_chunk_info(scene_idx, chunk_idx)
                
                if not scene or not chunk:
                    logger.error(f"Could not find scene {scene_idx+1} or chunk {chunk_idx+1}")
                    continue
                    
                num_frames_for_chunk = max(i2v_cfg.svd_min_frames if content_cfg.use_svd_flow else 8,
                                        int(chunk["target_duration"] * content_cfg.fps))
                gen_width, gen_height = content_cfg.generation_resolution
                
                if content_cfg.use_svd_flow:
                    # Generate keyframe image
                    keyframe_image_filename = f"scene_{scene_idx}_chunk_{chunk_idx}_keyframe.png"
                    keyframe_image_path = os.path.join(content_cfg.output_dir, keyframe_image_filename)
                    
                    t2i_module.generate_image(
                        visual_prompt, keyframe_image_path,
                        gen_width, gen_height, t2i_cfg
                    )
                    project_manager.update_chunk_status(scene_idx, chunk_idx, "image_generated",
                                                      keyframe_path=keyframe_image_path)
                    logger.info(f"Generated keyframe for scene {scene_idx+1} chunk {chunk_idx+1}")
                    
                    # Generate video from image
                    video_chunk_filename = f"scene_{scene_idx}_chunk_{chunk_idx}_svd.mp4"
                    video_chunk_path = os.path.join(content_cfg.output_dir, video_chunk_filename)
                    
                    sub_clip_path = i2v_module.generate_video_from_image(
                        keyframe_image_path, video_chunk_path,
                        num_frames_for_chunk, content_cfg.fps,
                        gen_width, gen_height, i2v_cfg,
                        motion_prompt=motion_prompt
                    )
                else:
                    video_chunk_filename = f"scene_{scene_idx}_chunk_{chunk_idx}_t2v.mp4"
                    video_chunk_path = os.path.join(content_cfg.output_dir, video_chunk_filename)
                    sub_clip_path = t2v_module.generate_video_from_text(
                        visual_prompt, video_chunk_path,
                        num_frames_for_chunk, content_cfg.fps,
                        gen_width, gen_height, t2v_cfg
                    )
                
                if sub_clip_path and os.path.exists(sub_clip_path):
                    project_manager.update_chunk_status(scene_idx, chunk_idx, "video_generated",
                                                      video_path=sub_clip_path)
                    project_manager.update_scene_status(scene_idx, "in_progress")
                    logger.info(f"Generated video for scene {scene_idx+1} chunk {chunk_idx+1}")
                else:
                    project_manager.update_chunk_status(scene_idx, chunk_idx, "failed")
                    logger.error(f"Failed to generate video for scene {scene_idx+1} chunk {chunk_idx+1}")
                
            elif task == "assemble_scene":
                scene_idx = task_data["scene_idx"]
                scene = project_manager.get_scene_info(scene_idx)
                
                if not scene:
                    logger.error(f"Could not find scene {scene_idx+1}")
                    continue
                    
                # Get all video chunks for the scene
                video_sub_clip_paths = [c["video_path"] for c in scene["chunks"] 
                                      if c["status"] == "video_generated"]
                
                if video_sub_clip_paths:
                    final_video_for_scene_path = assemble_scene_video_from_sub_clips(
                        video_sub_clip_paths, scene["narration"]["duration"], content_cfg, scene_idx
                    )
                    
                    if final_video_for_scene_path:
                        project_manager.update_scene_status(scene_idx, "completed",
                                                          assembled_video_path=final_video_for_scene_path)
                        logger.info(f"Assembled scene {scene_idx+1}")
                    else:
                        project_manager.update_scene_status(scene_idx, "failed")
                        logger.error(f"Failed to assemble scene {scene_idx+1}")
                else:
                    project_manager.update_scene_status(scene_idx, "failed")
                    logger.error(f"No video chunks found for scene {scene_idx+1}")
                
            elif task == "assemble_final":
                # Get all completed scenes
                processed_scene_assets = []
                for scene in project_manager.state.scenes:
                    if scene["status"] == "completed":
                        narration_data = {
                            "text": scene["narration"]["text"],
                            "duration": scene["narration"]["duration"]
                        }
                        processed_scene_assets.append((
                            scene["assembled_video_path"],
                            scene["narration"]["audio_path"],
                            narration_data
                        ))
                
                if processed_scene_assets:
                    final_video_path = assemble_final_reel(
                        processed_scene_assets, content_cfg,
                        output_filename=f"{topic.replace(' ','_').replace('.', '')}_final_reel.mp4"
                    )
                    
                    if final_video_path:
                        full_narration_text = " ".join([asset[2]["text"] for asset in processed_scene_assets])
                        project_manager.update_final_video(
                            final_video_path, "generated",
                            full_narration_text,
                            project_manager.state.script["hashtags"]
                        )
                        logger.info("Final video assembled successfully")
                    else:
                        project_manager.update_final_video("", "failed", "", [])
                        logger.error("Failed to assemble final video")
                else:
                    project_manager.update_final_video("", "failed", "", [])
                    logger.error("No completed scenes found for final assembly")
        
        # Clear TTS model after all scenes are processed
        tts_module.clear_tts_vram()
        
        if project_manager.is_completed():
            logger.info("\n--- AUTOMATION COMPLETE ---")
            logger.info(f"Final Video: {project_manager.state.final_video['path']}")
            logger.info(f"Suggested Caption Text:\n{project_manager.state.final_video['full_narration_text']}")
            logger.info(f"Suggested Hashtags: {', '.join(project_manager.state.final_video['hashtags'])}")
            final_video_path = project_manager.state.final_video['path']
        else:
            logger.warning("\n--- AUTOMATION FAILED OR NO OUTPUT ---")

    except Exception as e:
        logger.error(f"Unhandled error in main_automation_flow: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Final cleanup: Ensuring all modules attempt VRAM clearing...")
        # Call clear methods on all loaded module objects, they handle their own state
        if 'llm_module' in locals() and hasattr(llm_module, 'clear_llm_vram'): llm_module.clear_llm_vram()
        if 'tts_module' in locals() and hasattr(tts_module, 'clear_tts_vram'): tts_module.clear_tts_vram()
        if 't2i_module' in locals() and hasattr(t2i_module, 'clear_t2i_vram'): t2i_module.clear_t2i_vram()
        if 'i2v_module' in locals() and hasattr(i2v_module, 'clear_i2v_vram'): i2v_module.clear_i2v_vram()
        if 't2v_module' in locals() and hasattr(t2v_module, 'clear_t2v_vram'): t2v_module.clear_t2v_vram()
        
        # A final global clear just in case, though module clears should handle it
        clear_vram_globally() 
        logger.info("Cleanup finished.")
    
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
        generation_resolution=(576, 1024), # Optimal for SDXL and SVD-XT (9:16 for reels)
        final_output_resolution=(1080, 1920), # Instagram reels standard resolution
        output_dir="modular_reels_output/project3",
        font_for_subtitles="Arial" # Make sure this font is available or provide path to .ttf
    )

    module_choices = ModuleSelectorConfig() # Uses default modules and their configs

    speaker_audio_sample = "record_out.wav" # Path to your speaker reference WAV
    if not os.path.exists(speaker_audio_sample):
        print(f"Warning: Speaker ref audio '{speaker_audio_sample}' not found. XTTS uses default voice.")
        speaker_audio_sample = None
    
    reel_topic = "history of plantification in desert."
    
    start_time = time.time()
    generated_video = main_automation_flow(
        reel_topic, 
        content_settings, 
        module_choices,
        speaker_reference_audio=speaker_audio_sample,
        resume=False  # Set to True to resume from existing project
    )
    end_time = time.time()

    if generated_video:
        print(f"Successfully generated video: {generated_video}")
    else:
        print("Video generation failed or produced no output.")
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")