# In project_manager.py
import os
import logging
import shutil
from typing import Optional, Any
from uuid import UUID

from config_manager import ProjectState, Character, CharacterVersion, Voice, Scene, Shot

logger = logging.getLogger(__name__)

def _safe_remove(path: Optional[str]):
    if path and os.path.exists(path):
        try: os.remove(path)
        except OSError as e: logger.error(f"Error removing file {path}: {e}")

class ProjectManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir; self.project_file = os.path.join(output_dir, "project_state.json"); self.state: Optional[ProjectState] = None; os.makedirs(self.output_dir, exist_ok=True)
    def _save_state(self):
        if not self.state: return
        with open(self.project_file, 'w', encoding='utf-8') as f: f.write(self.state.model_dump_json(indent=2))
    def initialize_project(self, title: str, video_format: str):
        self.state = ProjectState(title=title, video_format=video_format); self._save_state()
    def load_project(self) -> bool:
        if not os.path.exists(self.project_file): return False
        try:
            with open(self.project_file, 'r', encoding='utf-8') as f: self.state = ProjectState.model_validate_json(f.read())
            needs_save = False
            if not hasattr(self.state, 'add_narration_text_to_video'): self.state.add_narration_text_to_video = True; needs_save = True
            for i, scene in enumerate(self.state.scenes):
                if not hasattr(scene, 'title'): scene.title = f"Scene {i + 1}"; needs_save = True
                for shot in scene.shots:
                    if not hasattr(shot, 'generation_flow'): shot.generation_flow = "T2I_I2V"; needs_save = True
                    if not hasattr(shot, 'uploaded_image_path'): shot.uploaded_image_path = None; needs_save = True
                    if not hasattr(shot, 'user_defined_duration'): shot.user_defined_duration = None; needs_save = True
            if needs_save: logger.warning(f"Upgraded old project format: '{self.state.title}'"); self._save_state()
            return True
        except Exception as e: logger.error(f"Error loading or migrating project: {e}", exc_info=True); return False
    def get_scene(self, uuid: UUID) -> Optional[Scene]: return next((s for s in self.state.scenes if s.uuid == uuid), None)
    def get_shot(self, scene_uuid: UUID, shot_uuid: UUID) -> Optional[Shot]:
        scene = self.get_scene(scene_uuid); return next((s for s in scene.shots if s.uuid == shot_uuid), None) if scene else None
    def get_character(self, uuid: UUID) -> Optional[Character]: return next((c for c in self.state.characters if c.uuid == uuid), None)
    def get_voice(self, uuid: UUID) -> Optional[Voice]: return next((v for v in self.state.voices if v.uuid == uuid), None)
    def _cleanup_shot_files(self, shot: Shot):
        _safe_remove(shot.keyframe_image_path); _safe_remove(shot.video_path); _safe_remove(shot.uploaded_image_path)
        shot_dir = os.path.join(self.output_dir, "shots", str(shot.uuid))
        if os.path.isdir(shot_dir):
            try: shutil.rmtree(shot_dir)
            except OSError as e: logger.error(f"Error removing shot directory {shot_dir}: {e}")
    def delete_scene(self, scene_uuid: UUID):
        if not self.state: return
        scene_to_delete = self.get_scene(scene_uuid)
        if scene_to_delete:
            for shot in scene_to_delete.shots: self._cleanup_shot_files(shot)
            _safe_remove(scene_to_delete.assembled_video_path); _safe_remove(scene_to_delete.narration.audio_path)
        self.state.scenes = [s for s in self.state.scenes if s.uuid != scene_uuid]; self._save_state()
    def delete_shot(self, scene_uuid: UUID, shot_uuid: UUID):
        scene = self.get_scene(scene_uuid)
        if scene:
            shot_to_delete = next((s for s in scene.shots if s.uuid == shot_uuid), None)
            if shot_to_delete: self._cleanup_shot_files(shot_to_delete)
            scene.shots = [s for s in scene.shots if s.uuid != shot_uuid]; self._save_state()
    def add_character(self, name: str, base_image_path: str):
        char = Character(name=name); version = CharacterVersion(name="base", reference_image_path=base_image_path); char.versions.append(version)
        self.state.characters.append(char); self._save_state()
    def add_voice(self, name: str, module_path: str, wav_path: str):
        voice = Voice(name=name, tts_module_path=module_path, reference_wav_path=wav_path); self.state.voices.append(voice); self._save_state()

    def add_scene(self, after_scene_uuid: Optional[UUID] = None):
        if not self.state: return
        new_scene = Scene(title=f"Scene {len(self.state.scenes) + 1}")
        if after_scene_uuid:
            try:
                target_index = next(i for i, s in enumerate(self.state.scenes) if s.uuid == after_scene_uuid)
                self.state.scenes.insert(target_index + 1, new_scene)
            except StopIteration:
                self.state.scenes.append(new_scene)
        else:
            self.state.scenes.append(new_scene)
        self._save_state()
        return new_scene.uuid

    def add_shot_to_scene(self, scene_uuid: UUID, after_shot_uuid: Optional[UUID] = None):
        scene = self.get_scene(scene_uuid)
        if scene:
            new_shot = Shot()
            if scene.shots:
                ref_shot_index = -1
                if after_shot_uuid:
                    try:
                        ref_shot_index = next(i for i, s in enumerate(scene.shots) if s.uuid == after_shot_uuid)
                    except StopIteration: pass
                
                ref_shot = scene.shots[ref_shot_index]
                new_shot.generation_flow = ref_shot.generation_flow
                new_shot.module_selections = ref_shot.module_selections.copy()

                if after_shot_uuid:
                    scene.shots.insert(ref_shot_index + 1, new_shot)
                else: 
                    scene.shots.insert(0, new_shot)
            else:
                 scene.shots.append(new_shot)
            self._save_state()
            return new_shot.uuid

    def reorder_scene(self, scene_uuid: UUID, direction: str):
        try:
            index = next(i for i, s in enumerate(self.state.scenes) if s.uuid == scene_uuid)
        except StopIteration: return
        if direction == 'up' and index > 0:
            self.state.scenes.insert(index - 1, self.state.scenes.pop(index))
        elif direction == 'down' and index < len(self.state.scenes) - 1:
            self.state.scenes.insert(index + 1, self.state.scenes.pop(index))
        self._save_state()

    def reorder_shot(self, scene_uuid: UUID, shot_uuid: UUID, direction: str):
        scene = self.get_scene(scene_uuid)
        if not scene: return
        try:
            index = next(i for i, s in enumerate(scene.shots) if s.uuid == shot_uuid)
        except StopIteration: return
        if direction == 'left' and index > 0:
            scene.shots.insert(index - 1, scene.shots.pop(index))
        elif direction == 'right' and index < len(scene.shots) - 1:
            scene.shots.insert(index + 1, scene.shots.pop(index))
        self._save_state()

    def update_scene(self, scene_uuid: UUID, data: dict):
        scene = self.get_scene(scene_uuid)
        if scene:
            if 'title' in data: scene.title = data['title']
            if 'narration_text' in data: scene.narration.text = data['narration_text']; scene.narration.status = "pending"; scene.narration.audio_path = None
            if 'voice_uuid' in data: scene.narration.voice_uuid = data['voice_uuid']; scene.narration.status = "pending"
            self._save_state()
            
    def update_shot(self, scene_uuid: UUID, shot_uuid: UUID, data: dict):
        shot = self.get_shot(scene_uuid, shot_uuid)
        if not shot: return

        # --- START: ROBUST INVALIDATION LOGIC ---
        # 1. Store the old state before any changes are made.
        old_state = {
            "generation_flow": shot.generation_flow,
            "visual_prompt": shot.visual_prompt,
            "motion_prompt": shot.motion_prompt,
            "module_selections": shot.module_selections.copy(),
            "character_uuids": shot.character_uuids.copy(),
            "user_defined_duration": shot.user_defined_duration
        }

        # 2. Apply all new data to the shot object.
        for key, value in data.items():
            if hasattr(shot, key):
                setattr(shot, key, value)
        
        # 3. Compare the new state against the old state to determine consequences.
        invalidates_image = False
        invalidates_video = False

        if shot.generation_flow != old_state["generation_flow"] or \
           shot.visual_prompt != old_state["visual_prompt"] or \
           shot.character_uuids != old_state["character_uuids"] or \
           shot.module_selections.get('t2i') != old_state["module_selections"].get('t2i'):
            invalidates_image = True
            
        if shot.motion_prompt != old_state["motion_prompt"] or \
           shot.user_defined_duration != old_state["user_defined_duration"] or \
           shot.module_selections.get('i2v') != old_state["module_selections"].get('i2v') or \
           shot.module_selections.get('t2v') != old_state["module_selections"].get('t2v'):
            invalidates_video = True

        # 4. Apply the invalidations.
        if invalidates_image:
            _safe_remove(shot.keyframe_image_path)
            shot.keyframe_image_path = None
            shot.status = "pending"
        
        # If image is invalidated, video MUST also be invalidated.
        if invalidates_image or invalidates_video:
            _safe_remove(shot.video_path)
            shot.video_path = None
            if shot.status != "pending":
                shot.status = "image_generated" if shot.keyframe_image_path and os.path.exists(shot.keyframe_image_path) else "pending"
        # --- END: ROBUST INVALIDATION LOGIC ---

        self._save_state()

    def update_final_video_path(self, path: Optional[str]):
        if self.state: self.state.final_video_path = path; self.state.status = "completed" if path else "in_progress"; self._save_state()