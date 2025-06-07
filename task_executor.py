#!/usr/bin/env python
# coding: utf-8

import logging
import math
import os
from typing import Optional, Dict, Any, Tuple

import torch
from project_manager import ProjectManager
from config_manager import ContentConfig, ModuleSelectorConfig
from video_assembly import assemble_final_reel, assemble_scene_video_from_sub_clips

logger = logging.getLogger(__name__)

class TaskExecutor:
    """
    Handles execution of individual tasks in the video generation pipeline.
    Each task can be executed independently and supports both automated and manual execution.
    """
    
    def __init__(self, project_manager: ProjectManager, content_cfg: ContentConfig, module_selector_cfg: ModuleSelectorConfig):
        self.project_manager = project_manager
        self.content_cfg = content_cfg
        self.module_selector_cfg = module_selector_cfg
        
        # Load modules
        self.llm_module = self._load_module(module_selector_cfg.llm_module)
        self.tts_module = self._load_module(module_selector_cfg.tts_module)
        self.t2i_module = self._load_module(module_selector_cfg.t2i_module)
        self.i2v_module = self._load_module(module_selector_cfg.i2v_module)
        self.t2v_module = self._load_module(module_selector_cfg.t2v_module)
        
        # Get configs
        self.llm_cfg = self.llm_module.LLMConfig()
        self.tts_cfg = self.tts_module.TTSConfig()
        self.t2i_cfg = self.t2i_module.T2IConfig()
        self.i2v_cfg = self.i2v_module.I2VConfig()
        self.t2v_cfg = self.t2v_module.T2VConfig()
    
    def _load_module(self, module_path_str: str):
        """Load a module dynamically."""
        try:
            module = __import__(module_path_str)
            logger.info(f"Successfully loaded module: {module_path_str}")
            return module
        except ImportError as e:
            logger.error(f"Error loading module {module_path_str}: {e}")
            raise
    
    def execute_task(self, task: str, task_data: Dict[str, Any], speaker_reference_audio: Optional[str] = None) -> bool:
        """
        Execute a single task in the pipeline.
        
        Args:
            task (str): Task name to execute
            task_data (Dict[str, Any]): Task-specific data
            speaker_reference_audio (Optional[str]): Path to speaker reference audio
            
        Returns:
            bool: True if task executed successfully, False otherwise
        """
        try:
            if task == "generate_script":
                return self._execute_generate_script(task_data["topic"])
            elif task == "generate_audio":
                # Use speaker_wav from task_data if provided, otherwise use speaker_reference_audio
                speaker_wav = task_data.get("speaker_wav", speaker_reference_audio)
                return self._execute_generate_audio(task_data["scene_idx"], task_data["text"], speaker_wav)
            elif task == "create_scene":
                return self._execute_create_scene(task_data["scene_idx"])
            elif task == "generate_chunk":
                return self._execute_generate_chunk(
                    task_data["scene_idx"],
                    task_data["chunk_idx"],
                    task_data["visual_prompt"],
                    task_data.get("motion_prompt")
                )
            elif task == "assemble_scene":
                return self._execute_assemble_scene(task_data["scene_idx"])
            elif task == "assemble_final":
                return self._execute_assemble_final()
            else:
                logger.error(f"Unknown task: {task}")
                return False
        except Exception as e:
            logger.error(f"Error executing task {task}: {e}")
            return False
    
    def _execute_generate_script(self, topic: str) -> bool:
        """Execute script generation task."""
        try:
            script_narration_parts, script_visual_prompts, hashtags = self.llm_module.generate_script(
                topic, self.content_cfg, self.llm_cfg
            )
            self.llm_module.clear_llm_vram()
            
            narration_parts = [{"text": part["text"], "status": "pending"} 
                             for part in script_narration_parts]
            visual_prompts = [{"prompt": prompt, "status": "pending"} 
                            for prompt in script_visual_prompts]
            self.project_manager.update_script(narration_parts, visual_prompts, hashtags)
            return True
        except Exception as e:
            logger.error(f"Error generating script: {e}")
            return False
    
    def _execute_generate_audio(self, scene_idx: int, text: str, speaker_wav: Optional[str] = None) -> bool:
        """Execute audio generation task."""
        try:
            scene_audio_path, actual_audio_duration = self.tts_module.generate_audio(
                text, self.content_cfg.output_dir, scene_idx, self.tts_cfg,
                speaker_wav=speaker_wav
            )
            
            if actual_audio_duration <= 0.1:
                logger.warning(f"Scene {scene_idx+1} has negligible audio.")
                return False
                
            narration_parts = self.project_manager.state.script["narration_parts"]
            narration_parts[scene_idx].update({
                "audio_path": scene_audio_path,
                "duration": actual_audio_duration,
                "status": "generated"
            })
            self.project_manager.update_script(
                narration_parts,
                self.project_manager.state.script["visual_prompts"],
                self.project_manager.state.script["hashtags"]
            )
            return True
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return False
    
    def _execute_create_scene(self, scene_idx: int) -> bool:
        """Execute scene creation task."""
        try:
            narration = self.project_manager.state.script["narration_parts"][scene_idx]
            visual_prompt = self.project_manager.state.script["visual_prompts"][scene_idx]
            
            num_video_chunks = math.ceil(narration["duration"] / self.content_cfg.model_max_video_chunk_duration)
            if num_video_chunks == 0:
                num_video_chunks = 1
                
            chunk_specific_prompts = self.llm_module.generate_chunk_visual_prompts(
                narration["text"], visual_prompt["prompt"], num_video_chunks, self.content_cfg, self.llm_cfg
            )
            self.llm_module.clear_llm_vram()
            
            chunks = []
            for chunk_idx in range(num_video_chunks):
                if chunk_idx < num_video_chunks - 1:
                    current_chunk_target_duration = self.content_cfg.model_max_video_chunk_duration
                else:
                    current_chunk_target_duration = narration["duration"] - (chunk_idx * self.content_cfg.model_max_video_chunk_duration)
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
            
            visual_prompts = self.project_manager.state.script["visual_prompts"]
            visual_prompts[scene_idx]["status"] = "generated"
            self.project_manager.update_script(
                self.project_manager.state.script["narration_parts"],
                visual_prompts,
                self.project_manager.state.script["hashtags"]
            )
            
            self.project_manager.add_scene(scene_idx, narration, chunks)
            return True
        except Exception as e:
            logger.error(f"Error creating scene: {e}")
            return False
    
    def _execute_generate_chunk(self, scene_idx: int, chunk_idx: int, visual_prompt: str, motion_prompt: Optional[str] = None) -> bool:
        """Execute chunk generation task."""
        try:
            scene = self.project_manager.get_scene_info(scene_idx)
            chunk = self.project_manager.get_chunk_info(scene_idx, chunk_idx)
            
            if not scene or not chunk:
                logger.error(f"Could not find scene {scene_idx+1} or chunk {chunk_idx+1}")
                return False
                
            num_frames_for_chunk = max(self.i2v_cfg.svd_min_frames if self.content_cfg.use_svd_flow else 8,
                                    int(chunk["target_duration"] * self.content_cfg.fps))
            gen_width, gen_height = self.content_cfg.generation_resolution
            
            if self.content_cfg.use_svd_flow:
                # Generate keyframe image
                keyframe_image_filename = f"scene_{scene_idx}_chunk_{chunk_idx}_keyframe.png"
                keyframe_image_path = os.path.join(self.content_cfg.output_dir, keyframe_image_filename)
                
                self.t2i_module.generate_image(
                    visual_prompt, keyframe_image_path,
                    gen_width, gen_height, self.t2i_cfg
                )
                self.project_manager.update_chunk_status(scene_idx, chunk_idx, "image_generated",
                                                      keyframe_path=keyframe_image_path)
                
                # Clear T2I VRAM before loading SVD
                self.t2i_module.clear_t2i_vram()
                torch.cuda.empty_cache()
                
                # Generate video from image
                video_chunk_filename = f"scene_{scene_idx}_chunk_{chunk_idx}_svd.mp4"
                video_chunk_path = os.path.join(self.content_cfg.output_dir, video_chunk_filename)
                
                sub_clip_path = self.i2v_module.generate_video_from_image(
                    keyframe_image_path, video_chunk_path,
                    num_frames_for_chunk, self.content_cfg.fps,
                    gen_width, gen_height, self.i2v_cfg,
                    motion_prompt=motion_prompt
                )
            else:
                video_chunk_filename = f"scene_{scene_idx}_chunk_{chunk_idx}_t2v.mp4"
                video_chunk_path = os.path.join(self.content_cfg.output_dir, video_chunk_filename)
                sub_clip_path = self.t2v_module.generate_video_from_text(
                    visual_prompt, video_chunk_path,
                    num_frames_for_chunk, self.content_cfg.fps,
                    gen_width, gen_height, self.t2v_cfg
                )
            
            if sub_clip_path and os.path.exists(sub_clip_path):
                self.project_manager.update_chunk_status(scene_idx, chunk_idx, "video_generated",
                                                      video_path=sub_clip_path)
                self.project_manager.update_scene_status(scene_idx, "in_progress")
                return True
            else:
                self.project_manager.update_chunk_status(scene_idx, chunk_idx, "failed")
                return False
        except Exception as e:
            logger.error(f"Error generating chunk: {e}")
            return False
    
    def _execute_assemble_scene(self, scene_idx: int) -> bool:
        """Execute scene assembly task."""
        try:
            scene = self.project_manager.get_scene_info(scene_idx)
            
            if not scene:
                logger.error(f"Could not find scene {scene_idx+1}")
                return False
                
            video_sub_clip_paths = [c["video_path"] for c in scene["chunks"] 
                                  if c["status"] == "video_generated"]
            
            if video_sub_clip_paths:
                final_video_for_scene_path = assemble_scene_video_from_sub_clips(
                    video_sub_clip_paths, scene["narration"]["duration"], self.content_cfg, scene_idx
                )
                
                if final_video_for_scene_path:
                    self.project_manager.update_scene_status(scene_idx, "completed",
                                                          assembled_video_path=final_video_for_scene_path)
                    return True
                else:
                    self.project_manager.update_scene_status(scene_idx, "failed")
                    return False
            else:
                self.project_manager.update_scene_status(scene_idx, "failed")
                return False
        except Exception as e:
            logger.error(f"Error assembling scene: {e}")
            return False
    
    def _execute_assemble_final(self) -> bool:
        """Execute final video assembly task."""
        try:
            processed_scene_assets = []
            for scene in self.project_manager.state.scenes:
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
                topic = self.project_manager.state.project_info["topic"]
                final_video_path = assemble_final_reel(
                    processed_scene_assets, self.content_cfg,
                    output_filename=f"{topic.replace(' ','_').replace('.', '')}_final_reel.mp4"
                )
                
                if final_video_path and os.path.exists(final_video_path):
                    full_narration_text = " ".join([asset[2]["text"] for asset in processed_scene_assets])
                    self.project_manager.update_final_video(
                        final_video_path, "generated",
                        full_narration_text,
                        self.project_manager.state.script["hashtags"]
                    )
                    return True
                else:
                    # Set status to pending instead of failed to allow retry
                    self.project_manager.update_final_video("", "pending", "", [])
                    return False
            else:
                # Set status to pending instead of failed to allow retry
                self.project_manager.update_final_video("", "pending", "", [])
                return False
        except Exception as e:
            logger.error(f"Error assembling final video: {e}")
            # Set status to pending instead of failed to allow retry
            self.project_manager.update_final_video("", "pending", "", [])
            return False 