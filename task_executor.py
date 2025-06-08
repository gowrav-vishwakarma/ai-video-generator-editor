# In task_executor.py

import logging
import math
import os
from typing import Optional, Dict, Any

import torch
from project_manager import ProjectManager
# Make sure these statuses are imported
from project_manager import STATUS_IMAGE_GENERATED, STATUS_VIDEO_GENERATED, STATUS_FAILED
from config_manager import ContentConfig, ModuleSelectorConfig
from video_assembly import assemble_final_reel, assemble_scene_video_from_sub_clips

logger = logging.getLogger(__name__)

class TaskExecutor:
    # ... (init, _load_module, and other execute methods remain the same) ...

    def __init__(self, project_manager: ProjectManager, content_cfg: ContentConfig, module_selector_cfg: ModuleSelectorConfig):
        self.project_manager = project_manager
        self.content_cfg = content_cfg
        self.module_selector_cfg = module_selector_cfg
        
        self.llm_module = self._load_module(module_selector_cfg.llm_module)
        self.tts_module = self._load_module(module_selector_cfg.tts_module)
        self.t2i_module = self._load_module(module_selector_cfg.t2i_module)
        self.i2v_module = self._load_module(module_selector_cfg.i2v_module)
        self.t2v_module = self._load_module(module_selector_cfg.t2v_module)
        
        self.llm_cfg = self.llm_module.LLMConfig()
        self.tts_cfg = self.tts_module.TTSConfig()
        self.t2i_cfg = self.t2i_module.T2IConfig()
        self.i2v_cfg = self.i2v_module.I2VConfig()
        self.t2v_cfg = self.t2v_module.T2VConfig()
    
    def _load_module(self, module_path_str: str):
        parts = module_path_str.split('.'); module_name = ".".join(parts)
        module = __import__(module_name, fromlist=[parts[-1]])
        logger.info(f"Successfully loaded module: {module_path_str}")
        return module

    def execute_task(self, task: str, task_data: Dict[str, Any]) -> bool:
        try:
            if task == "generate_script":
                return self._execute_generate_script(task_data["topic"])
            elif task == "generate_audio":
                return self._execute_generate_audio(task_data["scene_idx"], task_data["text"], task_data.get("speaker_wav"))
            elif task == "create_scene":
                return self._execute_create_scene(task_data["scene_idx"])
            elif task == "generate_chunk_image":
                return self._execute_generate_chunk_image(task_data["scene_idx"], task_data["chunk_idx"], task_data["visual_prompt"])
            elif task == "generate_chunk_video":
                return self._execute_generate_chunk_video(task_data["scene_idx"], task_data["chunk_idx"], task_data.get("motion_prompt"))
            elif task == "assemble_scene":
                return self._execute_assemble_scene(task_data["scene_idx"])
            elif task == "assemble_final":
                return self._execute_assemble_final()
            else:
                logger.error(f"Unknown task: {task}")
                return False
        except Exception as e:
            logger.error(f"Error executing task {task}: {e}", exc_info=True)
            return False

    def _execute_generate_script(self, topic: str) -> bool:
        try:
            script_narration_parts, script_visual_prompts, hashtags = self.llm_module.generate_script(
                topic, self.content_cfg, self.llm_cfg)
            self.llm_module.clear_llm_vram()
            
            if script_visual_prompts and isinstance(script_visual_prompts[0], str):
                 visual_prompts = [{"prompt": prompt, "status": "pending"} for prompt in script_visual_prompts]
            else:
                 visual_prompts = script_visual_prompts

            narration_parts = [{"text": part["text"], "status": "pending"} for part in script_narration_parts]
            self.project_manager.update_script(narration_parts, visual_prompts, hashtags)
            return True
        except Exception as e:
            logger.error(f"Error generating script: {e}"); return False

    def _execute_generate_audio(self, scene_idx: int, text: str, speaker_wav: Optional[str] = None) -> bool:
        try:
            scene_audio_path, duration = self.tts_module.generate_audio(text, self.content_cfg.output_dir, scene_idx, self.tts_cfg, speaker_wav=speaker_wav)
            if duration <= 0.1: logger.warning(f"Scene {scene_idx+1} has negligible audio."); duration = 0.0
            self.project_manager.update_narration_part_status(scene_idx, "generated", scene_audio_path, duration)
            return True
        except Exception as e:
            logger.error(f"Error generating audio: {e}"); return False

    def _execute_create_scene(self, scene_idx: int) -> bool:
        try:
            narration = self.project_manager.state.script["narration_parts"][scene_idx]
            visual_prompt_data = self.project_manager.state.script["visual_prompts"][scene_idx]
            num_chunks = math.ceil(narration["duration"] / self.content_cfg.model_max_video_chunk_duration)
            if num_chunks == 0: num_chunks = 1
            chunk_prompts = self.llm_module.generate_chunk_visual_prompts(
                narration["text"], visual_prompt_data["prompt"], num_chunks, self.content_cfg, self.llm_cfg)
            self.llm_module.clear_llm_vram()
            
            chunks = []
            for i in range(num_chunks):
                duration = self.content_cfg.model_max_video_chunk_duration if i < num_chunks - 1 else narration["duration"] - (i * self.content_cfg.model_max_video_chunk_duration)
                visual, motion = chunk_prompts[i]
                chunks.append({
                    "chunk_idx": i, "target_duration": max(0.5, duration),
                    "visual_prompt": visual, "motion_prompt": motion,
                    "keyframe_image_path": "", "video_path": "", "status": "pending"
                })
            self.project_manager.add_scene(scene_idx, chunks)
            return True
        except Exception as e:
            logger.error(f"Error creating scene: {e}"); return False

    def _execute_generate_chunk_image(self, scene_idx: int, chunk_idx: int, visual_prompt: str) -> bool:
        try:
            gen_width, gen_height = self.content_cfg.generation_resolution
            keyframe_filename = f"scene_{scene_idx}_chunk_{chunk_idx}_keyframe.png"
            keyframe_path = os.path.join(self.content_cfg.output_dir, keyframe_filename)
            
            self.t2i_module.generate_image(visual_prompt, keyframe_path, gen_width, gen_height, self.t2i_cfg)
            self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_IMAGE_GENERATED, keyframe_path=keyframe_path)
            self.t2i_module.clear_t2i_vram(); torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"Error generating chunk image: {e}")
            self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_FAILED)
            return False

    def _execute_generate_chunk_video(self, scene_idx: int, chunk_idx: int, motion_prompt: Optional[str] = None) -> bool:
        try:
            chunk = self.project_manager.get_scene_info(scene_idx)['chunks'][chunk_idx]
            keyframe_path = chunk.get("keyframe_image_path")
            if not keyframe_path or not os.path.exists(keyframe_path):
                logger.error(f"Cannot generate video for chunk {chunk_idx} in scene {scene_idx}: Keyframe image missing.")
                return False

            video_filename = f"scene_{scene_idx}_chunk_{chunk_idx}_svd.mp4"
            video_path = os.path.join(self.content_cfg.output_dir, video_filename)

            # --- THE CORRECTED CALL ---
            sub_clip_path = self.i2v_module.generate_video_from_image(
                image_path=keyframe_path, 
                output_video_path=video_path, 
                target_duration=chunk["target_duration"], 
                i2v_config=self.i2v_cfg, 
                motion_prompt=motion_prompt
            )
            
            if sub_clip_path and os.path.exists(sub_clip_path):
                self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_VIDEO_GENERATED, video_path=sub_clip_path)
                return True
            else:
                self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_FAILED)
                return False
        except Exception as e:
            logger.error(f"Error generating chunk video: {e}")
            self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_FAILED)
            return False

    def _execute_assemble_scene(self, scene_idx: int) -> bool:
        try:
            scene = self.project_manager.get_scene_info(scene_idx)
            if not scene: logger.error(f"Could not find scene {scene_idx}"); return False
            video_paths = [c["video_path"] for c in scene["chunks"] if c["status"] == STATUS_VIDEO_GENERATED]
            if len(video_paths) != len(scene['chunks']): logger.error(f"Not all chunks generated for scene {scene_idx}"); return False
            
            final_path = assemble_scene_video_from_sub_clips(video_paths, scene["narration"]["duration"], self.content_cfg, scene_idx)
            if final_path:
                self.project_manager.update_scene_status(scene_idx, "completed", assembled_video_path=final_path)
                return True
            self.project_manager.update_scene_status(scene_idx, "failed"); return False
        except Exception as e:
            logger.error(f"Error assembling scene: {e}"); self.project_manager.update_scene_status(scene_idx, "failed"); return False
    
    def _execute_assemble_final(self) -> bool:
        try:
            assets = []
            for scene in self.project_manager.state.scenes:
                if scene["status"] == "completed":
                    assets.append((scene["assembled_video_path"], scene["narration"]["audio_path"], {"text": scene["narration"]["text"], "duration": scene["narration"]["duration"]}))
            if len(assets) != len(self.project_manager.state.scenes): logger.error("Not all scenes completed"); return False

            topic = self.project_manager.state.project_info["topic"]
            final_path = assemble_final_reel(assets, self.content_cfg, output_filename=f"{topic.replace(' ','_')}_final.mp4")
            
            if final_path and os.path.exists(final_path):
                narration_text = " ".join([a[2]["text"] for a in assets])
                self.project_manager.update_final_video(final_path, "generated", narration_text, self.project_manager.state.script["hashtags"])
                return True
            self.project_manager.update_final_video("", "pending", "", []); return False
        except Exception as e:
            logger.error(f"Error assembling final video: {e}"); self.project_manager.update_final_video("", "pending", "", []); return False