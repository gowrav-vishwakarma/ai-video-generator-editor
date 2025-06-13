# In task_executor.py
import logging
import os
from uuid import UUID
from importlib import import_module
from typing import List
from project_manager import ProjectManager
from video_assembly import assemble_final_reel, assemble_scene_video_from_sub_clips
from config_manager import Shot, Scene, ContentConfig

logger = logging.getLogger(__name__)

def _import_class(module_path_str: str):
    if not module_path_str: return None
    try:
        module_path, class_name = module_path_str.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        print(f"Warning: Could not import module: {module_path_str}. Error: {e}")
        return None

class TaskExecutor:
    def __init__(self, project_manager: ProjectManager):
        self.pm = project_manager

    def get_module_max_duration(self, shot: Shot) -> float:
        flow = shot.generation_flow
        if flow in ("T2I_I2V", "Upload_I2V"): module_path = shot.module_selections.get('i2v')
        elif flow == "T2V": module_path = shot.module_selections.get('t2v')
        else: return 0.0
        if not module_path: return 0.0
        ModuleClass = _import_class(module_path)
        if not ModuleClass: return 0.0
        try: return ModuleClass.get_model_capabilities().get("max_shot_duration", 0.0)
        except Exception: return 0.0

    def calculate_shot_durations_for_scene(self, scene: Scene) -> List[float]:
        if not scene.narration.duration > 0 or not scene.shots:
            return [0.0] * len(scene.shots)
        remaining_time = scene.narration.duration
        calculated_durations = []
        for shot in scene.shots:
            if remaining_time <= 0:
                calculated_durations.append(0.0)
                continue
            module_capacity = self.get_module_max_duration(shot)
            if shot.user_defined_duration is not None and shot.user_defined_duration > 0:
                contribution = min(shot.user_defined_duration, module_capacity, remaining_time)
            else:
                contribution = min(module_capacity, remaining_time)
            calculated_durations.append(contribution)
            remaining_time -= contribution
        return calculated_durations

    def execute_task(self, task: str, task_data: dict) -> bool:
        task_map = {
            "generate_audio": self._execute_generate_audio,
            "generate_shot_image": self._execute_generate_shot_image,
            "generate_shot_video": self._execute_generate_shot_video,
            "generate_shot_t2v": self._execute_generate_shot_t2v,
            "assemble_final_video": self._execute_assemble_final,
        }
        try:
            if task in task_map: return task_map[task](**task_data)
            logger.error(f"Unknown task: {task}"); return False
        except Exception as e:
            logger.error(f"Error executing task {task}: {e}", exc_info=True); return False

    def _execute_generate_audio(self, scene_uuid: UUID) -> bool:
        scene = self.pm.get_scene(scene_uuid)
        if not scene or not scene.narration.voice_uuid: return False
        voice = self.pm.get_voice(scene.narration.voice_uuid)
        if not voice: return False
        try: scene_idx = self.pm.state.scenes.index(scene)
        except ValueError: logger.error(f"Could not find scene with UUID {scene_uuid} in project state."); return False
        TtsClass = _import_class(voice.tts_module_path); tts_module = TtsClass(TtsClass.Config())
        path, duration = tts_module.generate_audio(text=scene.narration.text, output_dir=self.pm.output_dir, scene_idx=scene_idx, language="en", speaker_wav=voice.reference_wav_path)
        tts_module.clear_vram()
        if path and os.path.exists(path):
            scene.narration.audio_path = path; scene.narration.duration = duration; scene.narration.status = "generated"; self.pm._save_state(); return True
        return False

    def _execute_generate_shot_image(self, scene_uuid: UUID, shot_uuid: UUID) -> bool:
        shot = self.pm.get_shot(scene_uuid, shot_uuid);
        if not shot: return False
        t2i_path = shot.module_selections.get('t2i');
        if not t2i_path: logger.error("T2I module not selected for T2I_I2V flow."); return False
        T2iClass = _import_class(t2i_path); t2i_module = T2iClass(T2iClass.Config())
        model_caps = t2i_module.get_model_capabilities(); w, h = model_caps["resolutions"][self.pm.state.video_format]
        path = os.path.join(self.pm.output_dir, "shots", f"shot_{shot.uuid}_keyframe.png"); os.makedirs(os.path.dirname(path), exist_ok=True)
        ip_adapter_paths = [self.pm.get_character(uid).versions[0].reference_image_path for uid in shot.character_uuids]
        img_path = t2i_module.generate_image(prompt=shot.visual_prompt, negative_prompt="bad quality", output_path=path, width=w, height=h, ip_adapter_image=ip_adapter_paths); t2i_module.clear_vram()
        if img_path: shot.keyframe_image_path = img_path; shot.status = "image_generated"; self.pm._save_state(); return True
        return False

    # --- START: DEFINITIVE FIX ---
    def _execute_generate_shot_video(self, scene_uuid: UUID, shot_uuid: UUID) -> bool:
        shot = self.pm.get_shot(scene_uuid, shot_uuid)
        if not shot: return False
        
        scene = self.pm.get_scene(scene_uuid)
        try: shot_index = scene.shots.index(shot)
        except ValueError: logger.error(f"Shot {shot_uuid} not found in scene {scene_uuid}"); return False
        
        all_durations = self.calculate_shot_durations_for_scene(scene)
        target_duration = all_durations[shot_index]
        
        if target_duration <= 0:
            logger.warning(f"Target duration for shot {shot.uuid} is 0. Skipping generation.")
            return True
        
        # --- UNCHANGED but now executed ---
        if shot.generation_flow == "Upload_I2V": shot.keyframe_image_path = shot.uploaded_image_path; self.pm._save_state()
        i2v_path = shot.module_selections.get('i2v')
        if not i2v_path: logger.error("I2V module not selected."); return False
        if not shot.keyframe_image_path or not os.path.exists(shot.keyframe_image_path): logger.error("Cannot generate video, keyframe image missing."); return False
        
        I2vClass = _import_class(i2v_path); i2v_module = I2vClass(I2vClass.Config())
        temp_config = ContentConfig(output_dir=self.pm.output_dir, aspect_ratio_format=self.pm.state.video_format)
        vid_path = os.path.join(self.pm.output_dir, "shots", f"shot_{shot.uuid}_video.mp4")
        
        result_path = i2v_module.generate_video_from_image(image_path=shot.keyframe_image_path, output_video_path=vid_path, target_duration=target_duration, content_config=temp_config, visual_prompt=shot.visual_prompt, motion_prompt=shot.motion_prompt); i2v_module.clear_vram()
        
        if result_path: shot.video_path = result_path; shot.status = "video_generated"; self.pm._save_state(); return True
        return False

    def _execute_generate_shot_t2v(self, scene_uuid: UUID, shot_uuid: UUID) -> bool:
        shot = self.pm.get_shot(scene_uuid, shot_uuid)
        if not shot: return False

        scene = self.pm.get_scene(scene_uuid)
        try: shot_index = scene.shots.index(shot)
        except ValueError: logger.error(f"Shot {shot_uuid} not found in scene {scene_uuid}"); return False

        all_durations = self.calculate_shot_durations_for_scene(scene)
        target_duration = all_durations[shot_index]

        if target_duration <= 0:
            logger.warning(f"Target duration for T2V shot {shot.uuid} is 0. Skipping generation.")
            return True
        
        # --- UNCHANGED but now executed ---
        t2v_path = shot.module_selections.get('t2v')
        if not t2v_path: logger.error("T2V module not selected for T2V flow."); return False
        
        T2vClass = _import_class(t2v_path); t2v_module = T2vClass(T2vClass.Config())
        model_caps = t2v_module.get_model_capabilities(); w, h = model_caps["resolutions"][self.pm.state.video_format]
        vid_path = os.path.join(self.pm.output_dir, "shots", f"shot_{shot.uuid}_video.mp4"); os.makedirs(os.path.dirname(vid_path), exist_ok=True)
        fps = 24; num_frames = int(target_duration * fps)
        
        ip_adapter_paths = [self.pm.get_character(uid).versions[0].reference_image_path for uid in shot.character_uuids]
        result_path = t2v_module.generate_video_from_text(prompt=shot.visual_prompt, output_video_path=vid_path, num_frames=num_frames, fps=fps, width=w, height=h, ip_adapter_image=ip_adapter_paths); t2v_module.clear_vram()
        
        if result_path: shot.video_path = result_path; shot.keyframe_image_path = None; shot.status = "video_generated"; self.pm._save_state(); return True
        return False
    # --- END: DEFINITIVE FIX ---
        
    def _execute_assemble_final(self, **kwargs) -> bool:
        assets = []
        for i, scene in enumerate(self.pm.state.scenes):
            shot_videos = [s.video_path for s in scene.shots if s.video_path and os.path.exists(s.video_path)]
            if not shot_videos or not scene.narration.audio_path or not os.path.exists(scene.narration.audio_path) or scene.narration.duration <= 0: continue
            
            scene_cfg = ContentConfig(output_dir=self.pm.output_dir)
            scene_video_path = assemble_scene_video_from_sub_clips(sub_clip_paths=shot_videos, target_total_duration=scene.narration.duration, config=scene_cfg, scene_idx=i)
            if not scene_video_path: continue
            scene.assembled_video_path = scene_video_path
            assets.append((scene.assembled_video_path, scene.narration.audio_path, {"text": scene.narration.text, "duration": scene.narration.duration}))
        
        if not assets: logger.error("No completed assets found to assemble."); return False
        self.pm._save_state()
        final_cfg = ContentConfig(output_dir=self.pm.output_dir, aspect_ratio_format=self.pm.state.video_format, add_narration_text_to_video=self.pm.state.add_narration_text_to_video)
        final_path = assemble_final_reel(assets, final_cfg)
        if final_path: self.pm.update_final_video_path(final_path); return True
        return False