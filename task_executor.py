# task_executor.py

import logging
import math
import os
from typing import Optional, Dict
import torch
from project_manager import ProjectManager, STATUS_IMAGE_GENERATED, STATUS_VIDEO_GENERATED, STATUS_FAILED
from config_manager import ContentConfig, ModuleSelectorConfig
from video_assembly import assemble_final_reel, assemble_scene_video_from_sub_clips

logger = logging.getLogger(__name__)

class TaskExecutor:
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
        
        # #############################################################################
        # # --- NEW DYNAMIC CONFIGURATION LOGIC ---
        # #############################################################################
        print("--- TaskExecutor: Configuring run from model capabilities... ---")
        
        # Determine which model's capabilities to use for video generation
        if self.content_cfg.use_svd_flow:
            # For T2I->I2V, resolution is from T2I, duration from I2V (or T2I as fallback)
            t2i_caps = self.t2i_module.get_model_capabilities()
            # In a full system, you might get this from i2v_module, but for now this is fine
            i2v_caps = self.i2v_module.get_model_capabilities() if hasattr(self.i2v_module, 'get_model_capabilities') else t2i_caps
            
            self.content_cfg.generation_resolution = t2i_caps["resolutions"].get(self.content_cfg.aspect_ratio_format)
            self.content_cfg.model_max_video_chunk_duration = i2v_caps.get("max_chunk_duration", 3.0)
        else:
            # For direct T2V, both resolution and duration come from the T2V model
            t2v_caps = self.t2v_module.get_model_capabilities()
            self.content_cfg.generation_resolution = t2v_caps["resolutions"].get(self.content_cfg.aspect_ratio_format)
            self.content_cfg.model_max_video_chunk_duration = t2v_caps.get("max_chunk_duration", 2.0)
            
        print(f"Dynamically set Generation Resolution to: {self.content_cfg.generation_resolution}")
        print(f"Dynamically set Max Chunk Duration to: {self.content_cfg.model_max_video_chunk_duration}s")
        print("-----------------------------------------------------------------")

        # Save the now-complete config back to the project file for consistency
        self.project_manager.state.project_info["config"] = self.content_cfg.__dict__
        self.project_manager._save_state()
        # #############################################################################

    def _load_module(self, module_path_str: str):
        parts = module_path_str.split('.'); module_name = ".".join(parts)
        return __import__(module_name, fromlist=[parts[-1]])

    def execute_task(self, task: str, task_data: Dict) -> bool:
        try:
            task_map = {
                "generate_script": self._execute_generate_script, "generate_audio": self._execute_generate_audio,
                "create_scene": self._execute_create_scene, "generate_chunk_image": self._execute_generate_chunk_image,
                "generate_chunk_video": self._execute_generate_chunk_video, "generate_chunk_t2v": self._execute_generate_chunk_t2v,
                "assemble_scene": self._execute_assemble_scene, "assemble_final": self._execute_assemble_final,
            }
            if task in task_map:
                return task_map[task](**task_data)
            logger.error(f"Unknown task: {task}"); return False
        except Exception as e:
            logger.error(f"Error executing task {task}: {e}", exc_info=True); return False

    def _execute_generate_script(self, topic: str) -> bool:
        narration_parts, visual_prompts, hashtags = self.llm_module.generate_script(topic, self.content_cfg, self.llm_cfg)
        self.llm_module.clear_llm_vram()
        narration_parts_with_status = [{"text": p["text"], "status": "pending"} for p in narration_parts]
        visual_prompts_with_status = [{"prompt": p, "status": "pending"} for p in visual_prompts]
        self.project_manager.update_script(narration_parts_with_status, visual_prompts_with_status, hashtags)
        return True

    def _execute_generate_audio(self, scene_idx: int, text: str, speaker_wav: Optional[str] = None) -> bool:
        path, duration = self.tts_module.generate_audio(text, self.content_cfg.output_dir, scene_idx, self.tts_cfg, speaker_wav=speaker_wav)
        if duration <= 0.1: duration = 0.0
        self.project_manager.update_narration_part_status(scene_idx, "generated", path, duration)
        return True

    def _execute_create_scene(self, scene_idx: int) -> bool:
        narration = self.project_manager.state.script["narration_parts"][scene_idx]
        visual_prompt_data = self.project_manager.state.script["visual_prompts"][scene_idx]
        num_chunks = math.ceil(narration["duration"] / self.content_cfg.model_max_video_chunk_duration)
        if num_chunks == 0: num_chunks = 1
        chunk_prompts = self.llm_module.generate_chunk_visual_prompts(narration["text"], visual_prompt_data["prompt"], num_chunks, self.content_cfg, self.llm_cfg)
        self.llm_module.clear_llm_vram()
        
        chunks = []
        for i in range(num_chunks):
            duration = self.content_cfg.model_max_video_chunk_duration if i < num_chunks - 1 else narration["duration"] - (i * self.content_cfg.model_max_video_chunk_duration)
            visual, motion = chunk_prompts[i]
            chunks.append({"chunk_idx": i, "target_duration": max(0.5, duration), "visual_prompt": visual, "motion_prompt": motion, "status": "pending"})
        self.project_manager.add_scene(scene_idx, chunks)
        return True

    def _execute_generate_chunk_image(self, scene_idx: int, chunk_idx: int, visual_prompt: str, **kwargs) -> bool:
        w, h = self.content_cfg.generation_resolution
        path = os.path.join(self.content_cfg.output_dir, f"scene_{scene_idx}_chunk_{chunk_idx}_keyframe.png")
        self.t2i_module.generate_image(visual_prompt, path, w, h, self.t2i_cfg)
        self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_IMAGE_GENERATED, keyframe_path=path)
        self.t2i_module.clear_t2i_vram()
        return True

    def _execute_generate_chunk_video(self, scene_idx: int, chunk_idx: int, motion_prompt: Optional[str], **kwargs) -> bool:
        chunk = self.project_manager.get_scene_info(scene_idx)['chunks'][chunk_idx]
        keyframe_path = chunk.get("keyframe_image_path")
        if not keyframe_path or not os.path.exists(keyframe_path): return False
        video_path = os.path.join(self.content_cfg.output_dir, f"scene_{scene_idx}_chunk_{chunk_idx}_svd.mp4")
        sub_clip_path = self.i2v_module.generate_video_from_image(keyframe_path, video_path, chunk["target_duration"], self.i2v_cfg, motion_prompt)
        if sub_clip_path and os.path.exists(sub_clip_path):
            self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_VIDEO_GENERATED, video_path=sub_clip_path)
            return True
        self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_FAILED); return False

    def _execute_generate_chunk_t2v(self, scene_idx: int, chunk_idx: int, visual_prompt: str, **kwargs) -> bool:
        chunk = self.project_manager.get_scene_info(scene_idx)['chunks'][chunk_idx]
        num_frames = int(chunk["target_duration"] * self.content_cfg.fps)
        w, h = self.content_cfg.generation_resolution
        video_path = os.path.join(self.content_cfg.output_dir, f"scene_{scene_idx}_chunk_{chunk_idx}_t2v.mp4")
        sub_clip_path = self.t2v_module.generate_video_from_text(visual_prompt, video_path, num_frames, self.content_cfg.fps, w, h, self.t2v_cfg)
        if sub_clip_path and os.path.exists(sub_clip_path):
            self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_VIDEO_GENERATED, video_path=sub_clip_path)
            return True
        self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_FAILED); return False

    def _execute_assemble_scene(self, scene_idx: int, **kwargs) -> bool:
        scene = self.project_manager.get_scene_info(scene_idx)
        if not scene: return False
        video_paths = [c["video_path"] for c in scene["chunks"] if c["status"] == STATUS_VIDEO_GENERATED]
        if len(video_paths) != len(scene['chunks']): return False
        final_path = assemble_scene_video_from_sub_clips(video_paths, scene["narration"]["duration"], self.content_cfg, scene_idx)
        if final_path:
            self.project_manager.update_scene_status(scene_idx, "completed", assembled_video_path=final_path)
            return True
        self.project_manager.update_scene_status(scene_idx, "failed"); return False
    
    def _execute_assemble_final(self, **kwargs) -> bool:
        assets = [(s["assembled_video_path"], s["narration"]["audio_path"], {"text": s["narration"]["text"], "duration": s["narration"]["duration"]}) for s in self.project_manager.state.scenes if s["status"] == "completed"]
        if len(assets) != len(self.project_manager.state.scenes): return False
        topic = self.project_manager.state.project_info["topic"]
        final_path = assemble_final_reel(assets, self.content_cfg, output_filename=f"{topic.replace(' ','_')}_final.mp4")
        if final_path and os.path.exists(final_path):
            text = " ".join([a[2]["text"] for a in assets])
            self.project_manager.update_final_video(final_path, "generated", text, self.project_manager.state.script["hashtags"])
            return True
        self.project_manager.update_final_video("", "pending", "", []); return False