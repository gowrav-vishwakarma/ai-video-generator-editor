# task_executor.py
import logging
import math
import os
from typing import Optional, Dict
import torch
from importlib import import_module

from project_manager import ProjectManager, STATUS_IMAGE_GENERATED, STATUS_VIDEO_GENERATED, STATUS_FAILED
from config_manager import ContentConfig, ModuleSelectorConfig
from video_assembly import assemble_final_reel, assemble_scene_video_from_sub_clips
from base_modules import BaseModuleConfig

logger = logging.getLogger(__name__)

def _import_class(module_path_str: str):
    module_path, class_name = module_path_str.rsplit('.', 1)
    module = import_module(module_path)
    return getattr(module, class_name)

class TaskExecutor:
    def __init__(self, project_manager: ProjectManager):
        self.project_manager = project_manager

        # --- Load configurations directly from the project state ---
        self.content_cfg = ContentConfig(**self.project_manager.state.project_info.config)
        module_selector_cfg = ModuleSelectorConfig()

        # --- Instantiate module classes ---
        # LLM
        LlmClass = _import_class(module_selector_cfg.llm_module)
        self.llm_cfg = LlmClass.Config()
        self.llm_module = LlmClass(self.llm_cfg)
        
        # TTS
        TtsClass = _import_class(module_selector_cfg.tts_module)
        self.tts_cfg = TtsClass.Config()
        self.tts_module = TtsClass(self.tts_cfg)

        # T2I
        T2iClass = _import_class(module_selector_cfg.t2i_module)
        self.t2i_cfg = T2iClass.Config()
        self.t2i_module = T2iClass(self.t2i_cfg)

        # I2V
        I2vClass = _import_class(module_selector_cfg.i2v_module)
        self.i2v_cfg = I2vClass.Config()
        self.i2v_module = I2vClass(self.i2v_cfg)

        # T2V
        T2vClass = _import_class(module_selector_cfg.t2v_module)
        self.t2v_cfg = T2vClass.Config()
        self.t2v_module = T2vClass(self.t2v_cfg)
        
        # --- Dynamic Configuration Logic ---
        self._configure_from_model_capabilities()

    def _configure_from_model_capabilities(self):
        print("--- TaskExecutor: Configuring run from model capabilities... ---")
        if self.content_cfg.use_svd_flow:
            t2i_caps = self.t2i_module.get_model_capabilities()
            i2v_caps = self.i2v_module.get_model_capabilities()
            self.content_cfg.generation_resolution = t2i_caps["resolutions"].get(self.content_cfg.aspect_ratio_format)
            self.content_cfg.model_max_video_chunk_duration = i2v_caps.get("max_chunk_duration", 3.0)
        else: # T2V Flow
            t2v_caps = self.t2v_module.get_model_capabilities()
            self.content_cfg.generation_resolution = t2v_caps["resolutions"].get(self.content_cfg.aspect_ratio_format)
            self.content_cfg.model_max_video_chunk_duration = t2v_caps.get("max_chunk_duration", 2.0)
            
        print(f"Dynamically set Generation Resolution to: {self.content_cfg.generation_resolution}")
        print(f"Dynamically set Max Chunk Duration to: {self.content_cfg.model_max_video_chunk_duration}s")
        print("-----------------------------------------------------------------")

        # Save the now-complete config back to the project file
        self.project_manager.state.project_info.config = self.content_cfg.model_dump()
        self.project_manager._save_state()

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
        narration_parts, visual_prompts, hashtags = self.llm_module.generate_script(topic, self.content_cfg)
        self.llm_module.clear_vram()
        narration_parts_with_status = [{"text": p["text"], "status": "pending"} for p in narration_parts]
        self.project_manager.update_script(narration_parts_with_status, visual_prompts, hashtags)
        return True

    def _execute_generate_audio(self, scene_idx: int, text: str, speaker_wav: Optional[str] = None) -> bool:
        path, duration = self.tts_module.generate_audio(text, self.content_cfg.output_dir, scene_idx, speaker_wav)
        if duration <= 0.1: duration = 0.0
        self.project_manager.update_narration_part_status(scene_idx, "generated", path, duration)
        return True

    def _execute_create_scene(self, scene_idx: int) -> bool:
        narration = self.project_manager.state.script.narration_parts[scene_idx]
        visual_prompt_data = self.project_manager.state.script.visual_prompts[scene_idx]
        num_chunks = math.ceil(narration.duration / self.content_cfg.model_max_video_chunk_duration)
        if num_chunks == 0: num_chunks = 1
        
        chunk_prompts = self.llm_module.generate_chunk_visual_prompts(narration.text, visual_prompt_data.prompt, num_chunks, self.content_cfg)
        self.llm_module.clear_vram()
        
        chunks = []
        for i in range(num_chunks):
            duration = self.content_cfg.model_max_video_chunk_duration if i < num_chunks - 1 else narration.duration - (i * self.content_cfg.model_max_video_chunk_duration)
            visual, motion = chunk_prompts[i]
            chunks.append({"chunk_idx": i, "target_duration": max(0.5, duration), "visual_prompt": visual, "motion_prompt": motion, "status": "pending"})
        self.project_manager.add_scene(scene_idx, chunks)
        return True

    def _execute_generate_chunk_image(self, scene_idx: int, chunk_idx: int, visual_prompt: str, **kwargs) -> bool:
        w, h = self.content_cfg.generation_resolution
        path = os.path.join(self.content_cfg.output_dir, f"scene_{scene_idx}_chunk_{chunk_idx}_keyframe.png")
        self.t2i_module.generate_image(visual_prompt, path, w, h)
        self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_IMAGE_GENERATED, keyframe_path=path)
        self.t2i_module.clear_vram()
        return True

    def _execute_generate_chunk_video(self, scene_idx: int, chunk_idx: int, visual_prompt: str, motion_prompt: Optional[str], **kwargs) -> bool:
        chunk = self.project_manager.get_scene_info(scene_idx).chunks[chunk_idx]
        if not chunk.keyframe_image_path or not os.path.exists(chunk.keyframe_image_path): return False
        
        video_path = os.path.join(self.content_cfg.output_dir, f"scene_{scene_idx}_chunk_{chunk_idx}_svd.mp4")
        sub_clip_path = self.i2v_module.generate_video_from_image(
            image_path=chunk.keyframe_image_path, 
            output_video_path=video_path, 
            target_duration=chunk.target_duration, 
            content_config=self.content_cfg,
            motion_prompt=motion_prompt, 
            visual_prompt=visual_prompt
        )
        if sub_clip_path and os.path.exists(sub_clip_path):
            self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_VIDEO_GENERATED, video_path=sub_clip_path)
            return True
        self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_FAILED); return False

    def _execute_generate_chunk_t2v(self, scene_idx: int, chunk_idx: int, visual_prompt: str, **kwargs) -> bool:
        chunk = self.project_manager.get_scene_info(scene_idx).chunks[chunk_idx]
        num_frames = int(chunk.target_duration * self.content_cfg.fps)
        w, h = self.content_cfg.generation_resolution
        video_path = os.path.join(self.content_cfg.output_dir, f"scene_{scene_idx}_chunk_{chunk_idx}_t2v.mp4")
        sub_clip_path = self.t2v_module.generate_video_from_text(visual_prompt, video_path, num_frames, self.content_cfg.fps, w, h)
        if sub_clip_path and os.path.exists(sub_clip_path):
            self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_VIDEO_GENERATED, video_path=sub_clip_path)
            return True
        self.project_manager.update_chunk_status(scene_idx, chunk_idx, STATUS_FAILED); return False

    def _execute_assemble_scene(self, scene_idx: int, **kwargs) -> bool:
        scene = self.project_manager.get_scene_info(scene_idx)
        if not scene: return False
        video_paths = [c.video_path for c in scene.chunks if c.status == STATUS_VIDEO_GENERATED]
        if len(video_paths) != len(scene.chunks): return False
        
        narration_duration = self.project_manager.state.script.narration_parts[scene_idx].duration
        final_path = assemble_scene_video_from_sub_clips(video_paths, narration_duration, self.content_cfg, scene_idx)
        
        if final_path:
            self.project_manager.update_scene_status(scene_idx, "completed", assembled_video_path=final_path)
            return True
        self.project_manager.update_scene_status(scene_idx, "failed"); return False
    
    def _execute_assemble_final(self, **kwargs) -> bool:
        narration_parts = self.project_manager.state.script.narration_parts
        assets = [
            (s.assembled_video_path, narration_parts[s.scene_idx].audio_path, {"text": narration_parts[s.scene_idx].text, "duration": narration_parts[s.scene_idx].duration})
            for s in self.project_manager.state.scenes if s.status == "completed"
        ]
        if len(assets) != len(self.project_manager.state.scenes): return False
        
        topic = self.project_manager.state.project_info.topic
        final_path = assemble_final_reel(assets, self.content_cfg, output_filename=f"{topic.replace(' ','_')}_final.mp4")
        
        if final_path and os.path.exists(final_path):
            text = " ".join([a[2]["text"] for a in assets])
            hashtags = self.project_manager.state.script.hashtags
            self.project_manager.update_final_video(final_path, "generated", text, hashtags)
            return True
        self.project_manager.update_final_video("", "pending", "", []); return False