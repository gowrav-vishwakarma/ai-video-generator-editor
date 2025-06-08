#!/usr/bin/env python
# coding: utf-8

# main_video_generator.py
import os
import importlib
import math
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
import torch

# Assume video_assembly.py is in the same directory or accessible in PYTHONPATH
from video_assembly import assemble_final_reel, assemble_scene_video_from_sub_clips
from config_manager import ContentConfig, ModuleSelectorConfig, DEVICE, clear_vram_globally
from moviepy import VideoFileClip # For getting duration
from project_manager import ProjectManager
from task_executor import TaskExecutor

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
    resume: bool = False,
    auto_mode: bool = True,
    start_from_task: Optional[str] = None
) -> Optional[str]:
    """
    Main workflow for video generation.
    
    Args:
        topic (str): Main topic for the video
        content_cfg (ContentConfig): Content configuration
        module_selector_cfg (ModuleSelectorConfig): Module selection configuration
        speaker_reference_audio (Optional[str]): Path to speaker reference audio
        resume (bool): Whether to resume an existing project
        auto_mode (bool): Whether to run in automated mode (True) or step-by-step mode (False)
        start_from_task (Optional[str]): Specific task to start from when resuming
        
    Returns:
        Optional[str]: Path to the final video if successful, None otherwise
    """
    final_video_path = None
    
    # Initialize project manager
    project_manager = ProjectManager(content_cfg.output_dir)
    
    if resume:
        if not project_manager.load_project():
            logger.warning("No existing project found to resume.")
            return None
        logger.info("Resuming existing project...")
        
        # If starting from a specific task, mark all tasks after it as pending
        if start_from_task:
            project_manager.mark_tasks_pending_after(start_from_task)
    else:
        project_manager.initialize_project(topic, content_cfg)
        logger.info("Initialized new project...")
    
    # Initialize task executor
    task_executor = TaskExecutor(project_manager, content_cfg, module_selector_cfg)
    
    try:
        while True:
            task, task_data = project_manager.get_next_pending_task()
            if not task:
                break
                
            logger.info(f"\n--- Processing task: {task} ---")
            
            if auto_mode:
                # Execute task automatically
                success = task_executor.execute_task(task, task_data, speaker_reference_audio)
                if not success:
                    logger.error(f"Task {task} failed")
                    break
            else:
                # In step-by-step mode, return the task for UI to handle
                return task, task_data, task_executor
        
        # Clear TTS model after all scenes are processed
        task_executor.tts_module.clear_tts_vram()
        
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
        # Call clear methods on all loaded module objects
        if hasattr(task_executor, 'llm_module'): task_executor.llm_module.clear_llm_vram()
        if hasattr(task_executor, 'tts_module'): task_executor.tts_module.clear_tts_vram()
        if hasattr(task_executor, 't2i_module'): task_executor.t2i_module.clear_t2i_vram()
        if hasattr(task_executor, 'i2v_module'): task_executor.i2v_module.clear_i2v_vram()
        if hasattr(task_executor, 't2v_module'): task_executor.t2v_module.clear_t2v_vram()
        
        # A final global clear just in case
        clear_vram_globally()
        logger.info("Cleanup finished.")
    
    return final_video_path

def execute_single_task(
    task: str,
    task_data: Dict[str, Any],
    task_executor: TaskExecutor,
    speaker_reference_audio: Optional[str] = None
) -> bool:
    """
    Execute a single task in the pipeline.
    
    Args:
        task (str): Task name to execute
        task_data (Dict[str, Any]): Task-specific data
        task_executor (TaskExecutor): Task executor instance
        speaker_reference_audio (Optional[str]): Path to speaker reference audio
        
    Returns:
        bool: True if task executed successfully, False otherwise
    """
    return task_executor.execute_task(task, task_data, speaker_reference_audio)

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
        resume=False,  # Set to True to resume from existing project
        auto_mode=True,  # Set to False for step-by-step mode
        start_from_task=None  # Set to a specific task if resuming
    )
    end_time = time.time()

    if generated_video:
        print(f"Successfully generated video: {generated_video}")
    else:
        print("Video generation failed or produced no output.")
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")