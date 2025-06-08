import os
import json
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
from config_manager import ContentConfig

# Configure logging
logger = logging.getLogger(__name__)

# --- STATUS DEFINITIONS ---
STATUS_PENDING = "pending"
STATUS_GENERATED = "generated"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_IN_PROGRESS = "in_progress"
STATUS_IMAGE_GENERATED = "image_generated" # New status for chunks
STATUS_VIDEO_GENERATED = "video_generated" # New status for chunks

@dataclass
class ProjectState:
    project_info: Dict[str, Any]
    script: Dict[str, Any]
    scenes: List[Dict[str, Any]] = field(default_factory=list)
    final_video: Dict[str, Any] = field(default_factory=dict)

class ProjectManager:
    """Manages the state and progression of a video generation project."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.project_file = os.path.join(output_dir, "project.json")
        self.state: Optional[ProjectState] = None
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _save_state(self):
        """Saves the current project state to project.json."""
        if not self.state: return
        self.state.project_info["last_modified"] = time.time()
        with open(self.project_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=4)
            
    def initialize_project(self, topic: str, config: ContentConfig):
        """Initializes a new project."""
        self.state = ProjectState(
            project_info={
                "topic": topic, "created_at": time.time(), "last_modified": time.time(),
                "status": STATUS_IN_PROGRESS, "config": asdict(config)
            },
            script={"narration_parts": [], "visual_prompts": [], "hashtags": []},
            scenes=[],
            final_video={"status": STATUS_PENDING}
        )
        self._save_state()
        logger.info(f"Initialized new project: {topic} in {self.output_dir}")
        
    def load_project(self) -> bool:
        """Loads project state from project.json."""
        if not os.path.exists(self.project_file):
            logger.error(f"Project file not found: {self.project_file}")
            return False
        try:
            with open(self.project_file, 'r') as f: data = json.load(f)
            data.setdefault('scenes', [])
            data.setdefault('final_video', {'status': STATUS_PENDING})
            if 'script' in data and 'visual_prompts' not in data['script']:
                data['script']['visual_prompts'] = []
            self.state = ProjectState(**data)
            logger.info(f"Loaded project: {self.state.project_info['topic']}")
            return True
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(f"Error loading project file {self.project_file}: {e}")
            return False

    def update_script(self, narration_parts: List[Dict], visual_prompts: List[Dict], hashtags: List[str]):
        if not self.state: return
        self.state.script["narration_parts"] = narration_parts
        self.state.script["visual_prompts"] = visual_prompts
        self.state.script["hashtags"] = hashtags
        self._save_state()

    def update_narration_part_text(self, part_idx: int, text: str):
        if not self.state or part_idx >= len(self.state.script["narration_parts"]): return
        part = self.state.script["narration_parts"][part_idx]
        if part["text"] != text:
            part["text"] = text; part["status"] = STATUS_PENDING; part["audio_path"] = ""; part["duration"] = 0
            self.state.scenes = [s for s in self.state.scenes if s.get('scene_idx') != part_idx]
            self._mark_final_for_reassembly()
            self._save_state()
            logger.info(f"Updated text for part {part_idx} and reset dependencies.")

    def add_scene(self, scene_idx: int, chunks: List[Dict]):
        if not self.state: return
        scene_data = {
            "scene_idx": scene_idx, "narration": self.state.script["narration_parts"][scene_idx],
            "chunks": chunks, "assembled_video_path": "", "status": STATUS_PENDING
        }
        self.state.scenes = [s for s in self.state.scenes if s.get('scene_idx') != scene_idx]
        self.state.scenes.append(scene_data)
        self.state.scenes.sort(key=lambda s: s['scene_idx'])
        self._save_state()

    def update_chunk_content(self, scene_idx: int, chunk_idx: int, visual_prompt: Optional[str] = None, motion_prompt: Optional[str] = None):
        scene = self.get_scene_info(scene_idx)
        if not scene or chunk_idx >= len(scene["chunks"]): return
        chunk = scene["chunks"][chunk_idx]
        changed = False
        if visual_prompt is not None and chunk["visual_prompt"] != visual_prompt:
            chunk["visual_prompt"] = visual_prompt; changed = True
        if motion_prompt is not None and chunk.get("motion_prompt", "") != motion_prompt:
            chunk["motion_prompt"] = motion_prompt; changed = True
        if changed:
            chunk["status"] = STATUS_PENDING; chunk["keyframe_image_path"] = ""; chunk["video_path"] = ""
            self._mark_scene_for_reassembly(scene_idx)
            self._save_state()
            logger.info(f"Updated prompts for chunk {chunk_idx} in scene {scene_idx} and reset it.")
            
    def _mark_scene_for_reassembly(self, scene_idx: int):
        scene = self.get_scene_info(scene_idx)
        if scene and scene["status"] == STATUS_COMPLETED:
            scene["status"] = STATUS_PENDING; scene["assembled_video_path"] = ""
            self._mark_final_for_reassembly()
            logger.info(f"Marked scene {scene_idx} for reassembly.")

    def _mark_final_for_reassembly(self):
        if self.state and self.state.final_video.get("status") == STATUS_GENERATED:
            self.state.final_video["status"] = STATUS_PENDING; self.state.final_video["path"] = ""
            self.state.project_info["status"] = STATUS_IN_PROGRESS
            logger.info("Marked final video for reassembly.")
    
    def get_next_pending_task(self) -> Tuple[Optional[str], Optional[Dict]]:
        if not self.state: return None, None
        if not self.state.script["narration_parts"]:
            return "generate_script", {"topic": self.state.project_info["topic"]}
        for i, part in enumerate(self.state.script["narration_parts"]):
            if part.get("status") != STATUS_GENERATED:
                return "generate_audio", {"scene_idx": i, "text": part["text"]}
        narration_indices_with_scenes = {s['scene_idx'] for s in self.state.scenes}
        for i, part in enumerate(self.state.script["narration_parts"]):
            if i not in narration_indices_with_scenes:
                return "create_scene", {"scene_idx": i}

        # #############################################################################
        # # --- CHANGE START: More specific chunk task identification ---
        # #############################################################################
        for scene in self.state.scenes:
            for chunk in scene["chunks"]:
                task_data = { "scene_idx": scene["scene_idx"], "chunk_idx": chunk["chunk_idx"],
                              "visual_prompt": chunk["visual_prompt"], "motion_prompt": chunk.get("motion_prompt")}
                
                # If chunk is pending, the next step is to generate its keyframe image
                if chunk.get("status") == STATUS_PENDING:
                    return "generate_chunk_image", task_data
                
                # If image is done but video is not, the next step is to generate the video
                if chunk.get("status") == STATUS_IMAGE_GENERATED:
                    return "generate_chunk_video", task_data
        # #############################################################################
        # # --- CHANGE END ---
        # #############################################################################

        for scene in self.state.scenes:
            all_chunks_done = all(c.get("status") == STATUS_VIDEO_GENERATED for c in scene["chunks"])
            if all_chunks_done and scene.get("status") != STATUS_COMPLETED:
                return "assemble_scene", {"scene_idx": scene["scene_idx"]}
        all_scenes_done = all(s.get("status") == STATUS_COMPLETED for s in self.state.scenes)
        if self.state.scenes and all_scenes_done and self.state.final_video.get("status") != STATUS_GENERATED:
            return "assemble_final", {}
        return None, None

    def get_scene_info(self, scene_idx: int) -> Optional[Dict]:
        if not self.state: return None
        return next((s for s in self.state.scenes if s["scene_idx"] == scene_idx), None)

    def update_narration_part_status(self, part_idx: int, status: str, audio_path: str = "", duration: float = 0.0):
        if not self.state or part_idx >= len(self.state.script["narration_parts"]): return
        part = self.state.script["narration_parts"][part_idx]
        part['status'] = status; part['audio_path'] = audio_path; part['duration'] = duration
        self._save_state()

    def update_chunk_status(self, scene_idx, chunk_idx, status, keyframe_path=None, video_path=None):
        scene = self.get_scene_info(scene_idx)
        if not scene or chunk_idx >= len(scene['chunks']): return
        chunk = scene['chunks'][chunk_idx]
        chunk['status'] = status
        if keyframe_path: chunk['keyframe_image_path'] = keyframe_path
        if video_path: chunk['video_path'] = video_path
        self._save_state()

    def update_scene_status(self, scene_idx, status, assembled_video_path=None):
        scene = self.get_scene_info(scene_idx)
        if not scene: return
        scene['status'] = status
        if assembled_video_path: scene['assembled_video_path'] = assembled_video_path
        self._save_state()
        
    def update_final_video(self, path, status, full_narration_text, hashtags):
        if not self.state: return
        self.state.final_video.update({
            "path": path, "status": status,
            "full_narration_text": full_narration_text, "hashtags": hashtags
        })
        if status == "generated":
            self.state.project_info["status"] = "completed"
        self._save_state()