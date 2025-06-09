# project_manager.py
import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field

from config_manager import ContentConfig

logger = logging.getLogger(__name__)

STATUS_PENDING, STATUS_GENERATED, STATUS_COMPLETED, STATUS_FAILED, STATUS_IN_PROGRESS = "pending", "generated", "completed", "failed", "in_progress"
STATUS_IMAGE_GENERATED, STATUS_VIDEO_GENERATED = "image_generated", "video_generated"

# --- Pydantic Models for Project State ---

class ProjectInfo(BaseModel):
    topic: str
    created_at: float = Field(default_factory=time.time)
    last_modified: float = Field(default_factory=time.time)
    status: str = STATUS_IN_PROGRESS
    config: Dict[str, Any]

class NarrationPart(BaseModel):
    text: str
    status: str = STATUS_PENDING
    audio_path: str = ""
    duration: float = 0.0

class VisualPrompt(BaseModel):
    prompt: str

class Script(BaseModel):
    # NEW: Fields for consistent context
    main_subject_description: str = ""
    setting_description: str = ""
    
    narration_parts: List[NarrationPart] = Field(default_factory=list)
    visual_prompts: List[VisualPrompt] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)

class Character(BaseModel):
    """Represents a character/subject in the project."""
    name: str
    reference_image_path: str
    source_prompt: Optional[str] = None
    source_image_path: Optional[str] = None # Path to the user-uploaded image


class Chunk(BaseModel):
    chunk_idx: int
    target_duration: float
    visual_prompt: str
    motion_prompt: Optional[str] = ""
    status: str = STATUS_PENDING
    keyframe_image_path: str = ""
    video_path: str = ""

class Scene(BaseModel):
    scene_idx: int
    status: str = STATUS_PENDING
    assembled_video_path: str = ""
    chunks: List[Chunk] = Field(default_factory=list)
    character_names: List[str] = Field(default_factory=list)

class FinalVideo(BaseModel):
    status: str = STATUS_PENDING
    path: str = ""
    full_narration_text: str = ""
    hashtags: List[str] = Field(default_factory=list)

class ProjectState(BaseModel):
    project_info: ProjectInfo
    script: Script = Field(default_factory=Script)
    characters: List[Character] = Field(default_factory=list)
    scenes: List[Scene] = Field(default_factory=list)
    final_video: FinalVideo = Field(default_factory=FinalVideo)

# --- ProjectManager Class ---

class ProjectManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.project_file = os.path.join(output_dir, "project.json")
        self.state: Optional[ProjectState] = None
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _save_state(self):
        if not self.state: return
        self.state.project_info.last_modified = time.time()
        with open(self.project_file, 'w') as f:
            f.write(self.state.model_dump_json(indent=4))
            
    def initialize_project(self, topic: str, config: ContentConfig):
        project_info = ProjectInfo(topic=topic, config=config.model_dump())
        self.state = ProjectState(project_info=project_info)
        self._save_state()
        
    def load_project(self) -> bool:
        if not os.path.exists(self.project_file): return False
        try:
            with open(self.project_file, 'r') as f:
                self.state = ProjectState.model_validate_json(f.read())
            return True
        except Exception as e:
            logger.error(f"Error loading project with Pydantic: {e}", exc_info=True); return False

    def update_script(self, script_data: Dict[str, Any]):
        if not self.state: return
        self.state.script.main_subject_description = script_data.get("main_subject_description", "")
        self.state.script.setting_description = script_data.get("setting_description", "")
        self.state.script.narration_parts = [NarrationPart(**p) for p in script_data.get("narration", [])]
        self.state.script.visual_prompts = [VisualPrompt(prompt=p) for p in script_data.get("visuals", [])]
        self.state.script.hashtags = script_data.get("hashtags", [])
        self._save_state()

    # ... get_next_pending_task and other methods remain the same ...
    def get_next_pending_task(self) -> Tuple[Optional[str], Optional[Dict]]:
        if not self.state: return None, None
        
        cfg = ContentConfig(**self.state.project_info.config)
        use_svd_flow = cfg.use_svd_flow

        if not self.state.script.narration_parts: return "generate_script", {"topic": self.state.project_info.topic}
        
        for i, part in enumerate(self.state.script.narration_parts):
            if part.status != STATUS_GENERATED: return "generate_audio", {"scene_idx": i, "text": part.text}
        
        narration_indices_with_scenes = {s.scene_idx for s in self.state.scenes}
        for i in range(len(self.state.script.narration_parts)):
            if i not in narration_indices_with_scenes: return "create_scene", {"scene_idx": i}

        for scene in sorted(self.state.scenes, key=lambda s: s.scene_idx):
            for chunk in sorted(scene.chunks, key=lambda c: c.chunk_idx):
                if chunk.status != STATUS_VIDEO_GENERATED:
                    task_data = { "scene_idx": scene.scene_idx, "chunk_idx": chunk.chunk_idx, "visual_prompt": chunk.visual_prompt, "motion_prompt": chunk.motion_prompt}
                    if use_svd_flow:
                        if chunk.status == STATUS_PENDING: return "generate_chunk_image", task_data
                        if chunk.status == STATUS_IMAGE_GENERATED: return "generate_chunk_video", task_data
                    else: # T2V Flow
                        return "generate_chunk_t2v", task_data

        for scene in self.state.scenes:
            if all(c.status == STATUS_VIDEO_GENERATED for c in scene.chunks) and scene.status != STATUS_COMPLETED:
                return "assemble_scene", {"scene_idx": scene.scene_idx}
        
        if self.state.scenes and all(s.status == STATUS_COMPLETED for s in self.state.scenes) and self.state.final_video.status != STATUS_GENERATED:
            return "assemble_final", {}
            
        return None, None

    def update_narration_part_text(self, part_idx: int, text: str):
        if not self.state or part_idx >= len(self.state.script.narration_parts): return
        part = self.state.script.narration_parts[part_idx]
        if part.text != text:
            part.text = text; part.status = STATUS_PENDING; part.audio_path = ""; part.duration = 0
            self.state.scenes = [s for s in self.state.scenes if s.scene_idx != part_idx]
            self._mark_final_for_reassembly()
            self._save_state()

    def add_scene(self, scene_idx: int, chunks: List[Dict]):
        if not self.state: return
        scene_data = Scene(scene_idx=scene_idx, chunks=[Chunk(**c) for c in chunks])
        self.state.scenes = [s for s in self.state.scenes if s.scene_idx != scene_idx]
        self.state.scenes.append(scene_data)
        self.state.scenes.sort(key=lambda s: s.scene_idx)
        self._save_state()

    def update_chunk_content(self, scene_idx: int, chunk_idx: int, visual_prompt: Optional[str] = None, motion_prompt: Optional[str] = None):
        scene = self.get_scene_info(scene_idx)
        if not scene or chunk_idx >= len(scene.chunks): return
        chunk = scene.chunks[chunk_idx]
        changed = False
        if visual_prompt is not None and chunk.visual_prompt != visual_prompt:
            chunk.visual_prompt = visual_prompt; changed = True
        if motion_prompt is not None and chunk.motion_prompt != motion_prompt:
            chunk.motion_prompt = motion_prompt; changed = True
        if changed:
            chunk.status = STATUS_PENDING; chunk.keyframe_image_path = ""; chunk.video_path = ""
            self._mark_scene_for_reassembly(scene_idx)
            self._save_state()
            
    def _mark_scene_for_reassembly(self, scene_idx: int):
        scene = self.get_scene_info(scene_idx)
        if scene and scene.status == STATUS_COMPLETED:
            scene.status = STATUS_PENDING; scene.assembled_video_path = ""
            self._mark_final_for_reassembly()

    def _mark_final_for_reassembly(self):
        if self.state and self.state.final_video.status == STATUS_GENERATED:
            self.state.final_video.status = STATUS_PENDING; self.state.final_video.path = ""
            self.state.project_info.status = STATUS_IN_PROGRESS
    
    def get_scene_info(self, scene_idx: int) -> Optional[Scene]:
        if not self.state: return None
        return next((s for s in self.state.scenes if s.scene_idx == scene_idx), None)

    def update_narration_part_status(self, part_idx: int, status: str, audio_path: str = "", duration: float = 0.0):
        if not self.state or part_idx >= len(self.state.script.narration_parts): return
        part = self.state.script.narration_parts[part_idx]
        part.status = status; part.audio_path = audio_path; part.duration = duration
        self._save_state()

    def update_chunk_status(self, scene_idx, chunk_idx, status, keyframe_path=None, video_path=None):
        scene = self.get_scene_info(scene_idx)
        if not scene or chunk_idx >= len(scene.chunks): return
        chunk = scene.chunks[chunk_idx]
        chunk.status = status
        if keyframe_path: chunk.keyframe_image_path = keyframe_path
        if video_path: chunk.video_path = video_path
        self._save_state()

    def update_scene_status(self, scene_idx, status, assembled_video_path=None):
        scene = self.get_scene_info(scene_idx)
        if not scene: return
        scene.status = status
        if assembled_video_path: scene.assembled_video_path = assembled_video_path
        self._save_state()
        
    def update_final_video(self, path, status, full_narration_text, hashtags):
        if not self.state: return
        self.state.final_video.path = path
        self.state.final_video.status = status
        self.state.final_video.full_narration_text = full_narration_text
        self.state.final_video.hashtags = hashtags
        if status == "generated": self.state.project_info.status = "completed"
        self._save_state()
    
    def add_character(self, character_data: Dict[str, Any]):
        if not self.state: return
        char = Character(**character_data)
        self.state.characters = [c for c in self.state.characters if c.name != char.name]
        self.state.characters.append(char)
        self._save_state()

    def update_character(self, old_name: str, new_name: str, new_reference_image_path: Optional[str]):
        char = self.get_character(old_name)
        if not char: return

        image_changed = new_reference_image_path and char.reference_image_path != new_reference_image_path
        name_changed = new_name and char.name != new_name

        if image_changed:
            char.reference_image_path = new_reference_image_path
            self._reset_visuals_for_character(old_name)

        if name_changed:
            char.name = new_name
            self._update_scene_references_on_name_change(old_name, new_name)
        
        self._save_state()

    def delete_character(self, name: str):
        if not self.state: return
        self.state.characters = [c for c in self.state.characters if c.name != name]
        for scene in self.state.scenes:
            if name in scene.character_names:
                scene.character_names.remove(name)
        
        safe_name = name.replace(" ", "_")
        char_dir = os.path.join(self.output_dir, "characters", safe_name)
        if os.path.exists(char_dir):
            shutil.rmtree(char_dir)
            print(f"Removed character asset directory: {char_dir}")

        self._save_state()
        
    def _reset_visuals_for_character(self, character_name: str):
        print(f"Resetting visuals for scenes containing character: {character_name}")
        for scene in self.state.scenes:
            if character_name in scene.character_names:
                for chunk in scene.chunks:
                    chunk.status = STATUS_PENDING
                    chunk.keyframe_image_path = ""
                    chunk.video_path = ""
                scene.status = STATUS_PENDING
                scene.assembled_video_path = ""
        self._mark_final_for_reassembly()
        
    def _update_scene_references_on_name_change(self, old_name: str, new_name: str):
        for scene in self.state.scenes:
            if old_name in scene.character_names:
                scene.character_names = [new_name if name == old_name else name for name in scene.character_names]

    def get_character(self, name: str) -> Optional[Character]:
        if not self.state: return None
        return next((c for c in self.state.characters if c.name == name), None)
        
    def update_scene_characters(self, scene_idx: int, character_names: List[str]):
        scene = self.get_scene_info(scene_idx)
        if scene:
            scene.character_names = character_names
            self._save_state()