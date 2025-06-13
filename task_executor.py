# In task_executor.py
import logging
import math
import os
import random
from typing import Optional, Dict
import torch
from importlib import import_module

from project_manager import ProjectManager, STATUS_IMAGE_GENERATED, STATUS_VIDEO_GENERATED, STATUS_FAILED
from config_manager import ContentConfig
from video_assembly import assemble_final_reel, assemble_scene_video_from_sub_clips
from base_modules import ModuleCapabilities

logger = logging.getLogger(__name__)

def _import_class(module_path_str: str):
    module_path, class_name = module_path_str.rsplit('.', 1)
    module = import_module(module_path)
    return getattr(module, class_name)

class TaskExecutor:
    def __init__(self, project_manager: ProjectManager):
        self.project_manager = project_manager
        self.content_cfg = ContentConfig(**self.project_manager.state.project_info.config)
        
        module_selections = self.content_cfg.module_selections
        if not module_selections:
            raise ValueError("Project state is missing module selections. Cannot initialize TaskExecutor.")

        # --- START OF FIX: Use .get() for safe module loading to prevent crashes ---
        
        # LLM and TTS are always required
        LlmClass = _import_class(module_selections["llm"])
        self.llm_module = LlmClass(LlmClass.Config())
        
        TtsClass = _import_class(module_selections["tts"])
        self.tts_module = TtsClass(TtsClass.Config())

        # Video modules are optional depending on the flow
        self.t2i_module = None
        self.i2v_module = None
        self.t2v_module = None
        
        t2i_path = module_selections.get("t2i")
        if t2i_path:
            T2iClass = _import_class(t2i_path)
            self.t2i_module = T2iClass(T2iClass.Config())

        i2v_path = module_selections.get("i2v")
        if i2v_path:
            I2vClass = _import_class(i2v_path)
            self.i2v_module = I2vClass(I2vClass.Config())

        t2v_path = module_selections.get("t2v")
        if t2v_path:
            T2vClass = _import_class(t2v_path)
            self.t2v_module = T2vClass(T2vClass.Config())

        # Determine capabilities based on which modules were actually loaded
        self.active_flow_supports_characters = False
        if self.content_cfg.use_svd_flow and self.t2i_module:
            t2i_caps = self.t2i_module.get_capabilities()
            self.active_flow_supports_characters = t2i_caps.supports_ip_adapter
            logger.info("Decisive module for character support: T2I module.")
        elif not self.content_cfg.use_svd_flow and self.t2v_module:
            t2v_caps = self.t2v_module.get_capabilities()
            self.active_flow_supports_characters = t2v_caps.supports_ip_adapter
            logger.info("Decisive module for character support: T2V module.")
        # --- END OF FIX ---
        
        logger.info(f"Holistic check: Active flow supports characters: {self.active_flow_supports_characters}")
        self._configure_from_model_capabilities()

    def _configure_from_model_capabilities(self):
        logger.info("--- TaskExecutor: Configuring run from model capabilities... ---")
        if self.content_cfg.use_svd_flow:
            if self.t2i_module and self.i2v_module:
                t2i_caps = self.t2i_module.get_model_capabilities()
                i2v_caps = self.i2v_module.get_model_capabilities()
                self.content_cfg.generation_resolution = t2i_caps["resolutions"].get(self.content_cfg.aspect_ratio_format)
                self.content_cfg.model_max_video_shot_duration = i2v_caps.get("max_shot_duration", 3.0)
            else:
                logger.warning("Warning: T2I or I2V module not loaded for I2V flow. Using default configurations.")
        else: # T2V Flow
            if self.t2v_module:
                t2v_caps = self.t2v_module.get_model_capabilities()
                self.content_cfg.generation_resolution = t2v_caps["resolutions"].get(self.content_cfg.aspect_ratio_format)
                self.content_cfg.model_max_video_shot_duration = t2v_caps.get("max_shot_duration", 2.0)
            else:
                logger.warning("Warning: T2V module not loaded for T2V flow. Using default configurations.")

        logger.info(f"Dynamically set Generation Resolution to: {self.content_cfg.generation_resolution}")
        logger.info(f"Dynamically set Max Shot Duration to: {self.content_cfg.model_max_video_shot_duration}s")
        self.project_manager.state.project_info.config = self.content_cfg.model_dump()
        self.project_manager._save_state()

    def execute_task(self, task: str, task_data: Dict) -> bool:
        try:
            # --- START OF FIX: Refresh config before every task to prevent stale state ---
            self.content_cfg = ContentConfig(**self.project_manager.state.project_info.config)
            logger.info(f"Executing task '{task}' with add_narration_text set to: {self.content_cfg.add_narration_text_to_video}")
            # --- END OF FIX ---
            
            task_map = {
                "generate_script": self._execute_generate_script, "generate_audio": self._execute_generate_audio,
                "create_scene": self._execute_create_scene, "generate_shot_image": self._execute_generate_shot_image,
                "generate_shot_video": self._execute_generate_shot_video, "generate_shot_t2v": self._execute_generate_shot_t2v,
                "assemble_scene": self._execute_assemble_scene, "assemble_final": self._execute_assemble_final,
            }
            if task in task_map: return task_map[task](**task_data)
            logger.error(f"Unknown task: {task}"); return False
        except Exception as e:
            logger.error(f"Error executing task {task}: {e}", exc_info=True); return False

    def _execute_generate_script(self, topic: str) -> bool:
        script_data = self.llm_module.generate_script(topic, self.content_cfg)
        self.llm_module.clear_vram()
        self.project_manager.update_script(script_data)
        return True

    def _execute_generate_audio(self, scene_idx: int, text: str, speaker_wav: Optional[str] = None) -> bool:
        path, duration = self.tts_module.generate_audio(text, self.content_cfg.output_dir, scene_idx, language=self.content_cfg.language, speaker_wav=speaker_wav)
        self.project_manager.update_narration_part_status(scene_idx, "generated", path, duration if duration > 0.1 else 0.0)
        return True

    def _execute_create_scene(self, scene_idx: int) -> bool:
        narration = self.project_manager.state.script.narration_parts[scene_idx]
        visual_prompt = self.project_manager.state.script.visual_prompts[scene_idx]
        main_subject = self.project_manager.state.script.main_subject_description
        setting = self.project_manager.state.script.setting_description
        
        actual_audio_duration = narration.duration
        max_shot_duration = self.content_cfg.model_max_video_shot_duration
        
        if actual_audio_duration <= 0 or max_shot_duration <= 0:
            num_shots = 1
            logger.warning(f"Warning: Invalid duration detected for Scene {scene_idx} (Audio: {actual_audio_duration}s, Max Shot: {max_shot_duration}s). Defaulting to 1 shot.")
        else:
            num_shots = math.ceil(actual_audio_duration / max_shot_duration) or 1

        logger.info("--- Calculating Shots for Scene {} ---".format(scene_idx))
        logger.info(f"  - Actual Audio Duration: {actual_audio_duration:.2f}s")
        logger.info(f"  - Model's Max Shot Duration: {max_shot_duration:.2f}s")
        logger.info(f"  - Calculated Number of Shots: {num_shots} ({actual_audio_duration:.2f}s / {max_shot_duration:.2f}s)")
        
        shot_prompts = self.llm_module.generate_shot_visual_prompts(
            narration.text, visual_prompt.prompt, num_shots, self.content_cfg, main_subject, setting
        )
        self.llm_module.clear_vram()
        
        shots = []
        for i, (visual, motion) in enumerate(shot_prompts):
            if i < num_shots - 1:
                duration = max_shot_duration
            else:
                duration = actual_audio_duration - (i * max_shot_duration)
            
            shots.append({"shot_idx": i, "target_duration": max(0.5, duration), "visual_prompt": visual, "motion_prompt": motion})
        
        all_character_names = [char.name for char in self.project_manager.state.characters]
        logger.info(f"Creating Scene {scene_idx} and assigning default characters: {all_character_names}")
        self.project_manager.add_scene(scene_idx, shots, character_names=all_character_names)
        return True

    def _execute_generate_shot_image(self, scene_idx: int, shot_idx: int, visual_prompt: str, **kwargs) -> bool:
        if not self.t2i_module:
            logger.error("Attempted to generate image, but T2I module is not loaded for this workflow.")
            return False
        w, h = self.content_cfg.generation_resolution
        path = os.path.join(self.content_cfg.output_dir, f"scene_{scene_idx}_shot_{shot_idx}_keyframe.png")
        
        base_seed = self.content_cfg.seed
        shot_seed = random.randint(0, 2**32 - 1) if base_seed == -1 else base_seed + scene_idx * 100 + shot_idx
        
        negative_prompt = "worst quality, low quality, bad anatomy, text, watermark, jpeg artifacts, blurry"
        
        scene = self.project_manager.get_scene_info(scene_idx)
        ip_adapter_image_paths = []
        if scene and scene.character_names:
            logger.info(f"Found characters for Scene {scene_idx}: {scene.character_names}")
            for name in scene.character_names:
                char = self.project_manager.get_character(name)
                if char and os.path.exists(char.reference_image_path):
                    ip_adapter_image_paths.append(char.reference_image_path)
        
        self.t2i_module.generate_image(
            prompt=visual_prompt, negative_prompt=negative_prompt, output_path=path, 
            width=w, height=h, ip_adapter_image=ip_adapter_image_paths or None, seed=shot_seed
        )
        
        self.project_manager.update_shot_status(scene_idx, shot_idx, STATUS_IMAGE_GENERATED, keyframe_path=path)
        self.t2i_module.clear_vram()
        return True

    def _execute_generate_shot_video(self, scene_idx: int, shot_idx: int, visual_prompt: str, motion_prompt: Optional[str], **kwargs) -> bool:
        if not self.i2v_module:
            logger.error("Attempted to generate video from image, but I2V module is not loaded for this workflow.")
            return False
        shot = self.project_manager.get_scene_info(scene_idx).shots[shot_idx]
        if not shot.keyframe_image_path or not os.path.exists(shot.keyframe_image_path): return False
        
        enhanced_visual = self.i2v_module.enhance_prompt(visual_prompt, "visual")
        enhanced_motion = self.i2v_module.enhance_prompt(motion_prompt, "motion")

        scene = self.project_manager.get_scene_info(scene_idx)
        ip_adapter_image_paths = [self.project_manager.get_character(name).reference_image_path for name in scene.character_names if self.project_manager.get_character(name)]

        video_path = os.path.join(self.content_cfg.output_dir, f"scene_{scene_idx}_shot_{shot_idx}_svd.mp4")
        
        sub_clip_path = self.i2v_module.generate_video_from_image(
            image_path=shot.keyframe_image_path, output_video_path=video_path, target_duration=shot.target_duration, 
            content_config=self.content_cfg, visual_prompt=enhanced_visual, motion_prompt=enhanced_motion,
            ip_adapter_image=ip_adapter_image_paths or None
        )

        if sub_clip_path and os.path.exists(sub_clip_path):
            self.project_manager.update_shot_status(scene_idx, shot_idx, STATUS_VIDEO_GENERATED, video_path=sub_clip_path)
            return True
        self.project_manager.update_shot_status(scene_idx, shot_idx, STATUS_FAILED); return False

    def _execute_generate_shot_t2v(self, scene_idx: int, shot_idx: int, visual_prompt: str, **kwargs) -> bool:
        if not self.t2v_module:
            logger.error("Attempted to generate video from text, but T2V module is not loaded for this workflow.")
            return False
        shot = self.project_manager.get_scene_info(scene_idx).shots[shot_idx]
        num_frames = int(shot.target_duration * self.content_cfg.fps)
        w, h = self.content_cfg.generation_resolution
        
        enhanced_prompt = self.t2v_module.enhance_prompt(visual_prompt)
        
        scene = self.project_manager.get_scene_info(scene_idx)
        ip_adapter_image_paths = [self.project_manager.get_character(name).reference_image_path for name in scene.character_names if self.project_manager.get_character(name)]
        
        video_path = os.path.join(self.content_cfg.output_dir, f"scene_{scene_idx}_shot_{shot_idx}_t2v.mp4")
        
        sub_clip_path = self.t2v_module.generate_video_from_text(
            enhanced_prompt, video_path, num_frames, self.content_cfg.fps, w, h,
            ip_adapter_image=ip_adapter_image_paths or None
        )

        if sub_clip_path and os.path.exists(sub_clip_path):
            self.project_manager.update_shot_status(scene_idx, shot_idx, STATUS_VIDEO_GENERATED, video_path=sub_clip_path)
            return True
        self.project_manager.update_shot_status(scene_idx, shot_idx, STATUS_FAILED); return False

    def _execute_assemble_scene(self, scene_idx: int, **kwargs) -> bool:
        scene = self.project_manager.get_scene_info(scene_idx)
        if not scene: return False
        video_paths = [c.video_path for c in scene.shots if c.status == STATUS_VIDEO_GENERATED]
        if len(video_paths) != len(scene.shots): return False
        
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