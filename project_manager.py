import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from config_manager import ContentConfig

@dataclass
class ProjectState:
    """Represents the state of a video generation project."""
    project_info: Dict[str, Any]
    script: Dict[str, Any]
    scenes: List[Dict[str, Any]]
    final_video: Dict[str, Any]

class ProjectManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.project_file = os.path.join(output_dir, "project.json")
        self.state: Optional[ProjectState] = None
        
    def initialize_project(self, topic: str, config: ContentConfig) -> None:
        """Initialize a new project with the given topic and config."""
        self.state = ProjectState(
            project_info={
                "topic": topic,
                "created_at": time.time(),
                "last_modified": time.time(),
                "status": "in_progress",
                "config": asdict(config)
            },
            script={
                "narration_parts": [],
                "visual_prompts": [],
                "hashtags": []
            },
            scenes=[],
            final_video={
                "path": "",
                "status": "pending",
                "full_narration_text": "",
                "hashtags": []
            }
        )
        self._save_state()
        
    def load_project(self) -> bool:
        """Load project state from project.json. Returns True if successful."""
        if not os.path.exists(self.project_file):
            return False
            
        try:
            with open(self.project_file, 'r') as f:
                data = json.load(f)
                self.state = ProjectState(**data)
            return True
        except Exception as e:
            print(f"Error loading project: {e}")
            return False
            
    def _save_state(self) -> None:
        """Save current project state to project.json."""
        if not self.state:
            return
            
        self.state.project_info["last_modified"] = time.time()
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(self.project_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)
            
    def update_script(self, narration_parts: List[Dict], visual_prompts: List[Dict], hashtags: List[str]) -> None:
        """Update script information in project state."""
        if not self.state:
            return
            
        self.state.script["narration_parts"] = narration_parts
        self.state.script["visual_prompts"] = visual_prompts
        self.state.script["hashtags"] = hashtags
        self._save_state()
        
    def add_scene(self, scene_idx: int, narration: Dict, chunks: List[Dict]) -> None:
        """Add or update a scene in the project state."""
        if not self.state:
            return
            
        # Find existing scene
        existing_scene_idx = next((i for i, s in enumerate(self.state.scenes) 
                                 if s["scene_idx"] == scene_idx), None)
        
        if existing_scene_idx is not None:
            # Update existing scene
            self.state.scenes[existing_scene_idx].update({
                "narration": narration,
                "chunks": chunks,
                "status": "pending"
            })
        else:
            # Add new scene
            new_scene = {
                "scene_idx": scene_idx,
                "narration": narration,
                "chunks": chunks,
                "assembled_video_path": "",
                "status": "pending"
            }
            self.state.scenes.append(new_scene)
            
        self._save_state()
        
    def update_chunk_status(self, scene_idx: int, chunk_idx: int, status: str, 
                          keyframe_path: Optional[str] = None, video_path: Optional[str] = None) -> None:
        """Update status and paths for a specific chunk."""
        if not self.state:
            return
            
        scene = next((s for s in self.state.scenes if s["scene_idx"] == scene_idx), None)
        if not scene:
            return
            
        chunk = next((c for c in scene["chunks"] if c["chunk_idx"] == chunk_idx), None)
        if not chunk:
            return
            
        chunk["status"] = status
        if keyframe_path:
            chunk["keyframe_image_path"] = keyframe_path
        if video_path:
            chunk["video_path"] = video_path
            
        self._save_state()
        
    def update_scene_status(self, scene_idx: int, status: str, assembled_video_path: Optional[str] = None) -> None:
        """Update status and assembled video path for a scene."""
        if not self.state:
            return
            
        scene = next((s for s in self.state.scenes if s["scene_idx"] == scene_idx), None)
        if not scene:
            return
            
        scene["status"] = status
        if assembled_video_path:
            scene["assembled_video_path"] = assembled_video_path
            
        self._save_state()
        
    def update_final_video(self, path: str, status: str, full_narration_text: str, hashtags: List[str]) -> None:
        """Update final video information."""
        if not self.state:
            return
            
        self.state.final_video.update({
            "path": path,
            "status": status,
            "full_narration_text": full_narration_text,
            "hashtags": hashtags
        })
        
        if status == "generated":
            self.state.project_info["status"] = "completed"
            
        self._save_state()
        
    def get_next_pending_task(self) -> Tuple[str, Dict]:
        """Get the next pending task in the generation pipeline."""
        if not self.state:
            return None, None
            
        # Check script generation
        if not self.state.script["narration_parts"]:
            return "generate_script", {"topic": self.state.project_info["topic"]}
            
        # Check audio generation
        for i, narration in enumerate(self.state.script["narration_parts"]):
            if narration.get("status") != "generated":
                return "generate_audio", {"scene_idx": i, "text": narration["text"]}
                
        # Check scene creation for each narration part
        for i, narration in enumerate(self.state.script["narration_parts"]):
            # Find if scene exists for this narration
            scene_exists = any(s["scene_idx"] == i for s in self.state.scenes)
            if narration.get("status") == "generated" and narration.get("audio_path") and not scene_exists:
                return "create_scene", {"scene_idx": i}
                
        # Check video generation for each scene
        for scene in self.state.scenes:
            # Check chunks that need generation
            for chunk in scene["chunks"]:
                if chunk["status"] == "pending":
                    return "generate_chunk", {
                        "scene_idx": scene["scene_idx"],
                        "chunk_idx": chunk["chunk_idx"],
                        "visual_prompt": chunk["visual_prompt"],
                        "motion_prompt": chunk.get("motion_prompt")
                    }
                    
            # Check if scene needs assembly
            if (scene["status"] in ["pending", "in_progress"]) and all(c["status"] == "video_generated" for c in scene["chunks"]):
                return "assemble_scene", {"scene_idx": scene["scene_idx"]}
                
        # Check if final video needs assembly
        if all(s["status"] == "completed" for s in self.state.scenes) and self.state.final_video["status"] != "generated":
            return "assemble_final", {}
            
        return None, None
        
    def get_scene_info(self, scene_idx: int) -> Optional[Dict]:
        """Get information about a specific scene."""
        if not self.state:
            return None
            
        return next((s for s in self.state.scenes if s["scene_idx"] == scene_idx), None)
        
    def get_chunk_info(self, scene_idx: int, chunk_idx: int) -> Optional[Dict]:
        """Get information about a specific chunk."""
        scene = self.get_scene_info(scene_idx)
        if not scene:
            return None
            
        return next((c for c in scene["chunks"] if c["chunk_idx"] == chunk_idx), None)
        
    def is_completed(self) -> bool:
        """Check if the project is completed."""
        if not self.state:
            return False
            
        return self.state.project_info["status"] == "completed" 