"""
Project Manager Module for Video Generation Pipeline

This module implements a state management system for a video generation pipeline. It tracks the progress
of video generation tasks, manages dependencies between different stages, and ensures proper regeneration
of affected components when changes are made.

The system breaks down video generation into discrete tasks:
1. Script Generation: Creates narration parts and visual prompts
2. Audio Generation: Generates audio for each narration part
3. Scene Creation: Creates scenes with video chunks based on narration duration
4. Chunk Generation: Generates individual video chunks for each scene
5. Scene Assembly: Combines chunks into complete scenes
6. Final Assembly: Combines all scenes into the final video

The ProjectManager maintains a project.json file that tracks:
- Project metadata and configuration
- Script content (narration parts and visual prompts)
- Scene information and their chunks
- Final video status and metadata

Key Features:
- State Tracking: Each component (script, audio, video chunks, scenes) has clear status tracking
- Dependency Management: Changes to components trigger appropriate regeneration of dependent parts
- Resumability: Projects can be saved and resumed at any point
- Selective Regeneration: Only affected components are regenerated when changes are made
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from config_manager import ContentConfig

@dataclass
class ProjectState:
    """
    Represents the complete state of a video generation project.
    
    Attributes:
        project_info (Dict[str, Any]): Project metadata including topic, timestamps, and configuration
        script (Dict[str, Any]): Script content including narration parts and visual prompts
        scenes (List[Dict[str, Any]]): List of scenes, each containing narration and video chunks
        final_video (Dict[str, Any]): Information about the final assembled video
    """
    project_info: Dict[str, Any]
    script: Dict[str, Any]
    scenes: List[Dict[str, Any]]
    final_video: Dict[str, Any]

class ProjectManager:
    """
    Manages the state and progression of a video generation project.
    
    The ProjectManager handles:
    - Project initialization and state persistence
    - Task scheduling and dependency management
    - Content updates and regeneration triggers
    - Progress tracking and status updates
    
    The project state is stored in a project.json file in the specified output directory.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the ProjectManager.
        
        Args:
            output_dir (str): Directory where project files will be stored
        """
        self.output_dir = output_dir
        self.project_file = os.path.join(output_dir, "project.json")
        self.state: Optional[ProjectState] = None
        
    def initialize_project(self, topic: str, config: ContentConfig) -> None:
        """
        Initialize a new project with the given topic and configuration.
        
        Creates a new project state with:
        - Project metadata (topic, timestamps, config)
        - Empty script structure
        - Empty scenes list
        - Pending final video status
        
        Args:
            topic (str): The main topic for the video
            config (ContentConfig): Configuration settings for the project
        """
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
        """
        Load project state from project.json.
        
        Also checks for any non-generated chunks and triggers the dependency chain
        to ensure proper regeneration of affected components.
        
        Returns:
            bool: True if project was successfully loaded, False otherwise
        """
        if not os.path.exists(self.project_file):
            return False
            
        try:
            with open(self.project_file, 'r') as f:
                data = json.load(f)
                self.state = ProjectState(**data)
                
            # Check for any non-generated chunks and trigger dependency chain
            for scene in self.state.scenes:
                for chunk in scene["chunks"]:
                    if chunk["status"] != "generated":
                        self._mark_scene_for_reassembly(scene["scene_idx"])
                        self._mark_final_for_reassembly()
                        break
                        
            return True
        except Exception as e:
            print(f"Error loading project: {e}")
            return False
            
    def _save_state(self) -> None:
        """
        Save current project state to project.json.
        
        Updates the last_modified timestamp before saving.
        Creates the output directory if it doesn't exist.
        """
        if not self.state:
            return
            
        self.state.project_info["last_modified"] = time.time()
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(self.project_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)
            
    def update_script(self, narration_parts: List[Dict], visual_prompts: List[Dict], hashtags: List[str]) -> None:
        """
        Update script information in project state.
        
        Args:
            narration_parts (List[Dict]): List of narration parts with text and status
            visual_prompts (List[Dict]): List of visual prompts with prompt text and status
            hashtags (List[str]): List of hashtags for the video
        """
        if not self.state:
            return
            
        self.state.script["narration_parts"] = narration_parts
        self.state.script["visual_prompts"] = visual_prompts
        self.state.script["hashtags"] = hashtags
        self._save_state()
        
    def add_scene(self, scene_idx: int, narration: Dict, chunks: List[Dict]) -> None:
        """
        Add or update a scene in the project state.
        
        If a scene with the given index exists, it will be updated.
        Otherwise, a new scene will be added.
        
        Args:
            scene_idx (int): Index of the scene
            narration (Dict): Narration information for the scene
            chunks (List[Dict]): List of video chunks for the scene
        """
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
        
    def _mark_scene_for_reassembly(self, scene_idx: int) -> None:
        """
        Mark a scene for reassembly and update its status.
        
        Clears the assembled video path and sets status to pending.
        
        Args:
            scene_idx (int): Index of the scene to mark for reassembly
        """
        if not self.state:
            return
            
        scene = next((s for s in self.state.scenes if s["scene_idx"] == scene_idx), None)
        if scene:
            scene["status"] = "pending"
            scene["assembled_video_path"] = ""
            self._save_state()
            
    def _mark_final_for_reassembly(self) -> None:
        """
        Mark final video for reassembly.
        
        Clears the final video path and resets its status to pending.
        """
        if not self.state:
            return
            
        self.state.final_video.update({
            "path": "",
            "status": "pending",
            "full_narration_text": "",
            "hashtags": []
        })
        self._save_state()
        
    def update_chunk_status(self, scene_idx: int, chunk_idx: int, status: str, 
                          keyframe_path: Optional[str] = None, video_path: Optional[str] = None) -> None:
        """
        Update status and paths for a specific chunk.
        
        If the chunk status changes from generated to something else,
        triggers the dependency chain to reassemble affected components.
        
        Args:
            scene_idx (int): Index of the scene containing the chunk
            chunk_idx (int): Index of the chunk to update
            status (str): New status for the chunk
            keyframe_path (Optional[str]): Path to the keyframe image if available
            video_path (Optional[str]): Path to the generated video if available
        """
        if not self.state:
            return
            
        scene = next((s for s in self.state.scenes if s["scene_idx"] == scene_idx), None)
        if not scene:
            return
            
        chunk = next((c for c in scene["chunks"] if c["chunk_idx"] == chunk_idx), None)
        if not chunk:
            return
            
        # If chunk status is changing from generated to something else
        if chunk["status"] == "generated" and status != "generated":
            # Mark scene for reassembly
            self._mark_scene_for_reassembly(scene_idx)
            # Mark final video for reassembly
            self._mark_final_for_reassembly()
            
        chunk["status"] = status
        if keyframe_path:
            chunk["keyframe_image_path"] = keyframe_path
        if video_path:
            chunk["video_path"] = video_path
            
        self._save_state()
        
    def update_scene_status(self, scene_idx: int, status: str, assembled_video_path: Optional[str] = None) -> None:
        """
        Update status and assembled video path for a scene.
        
        Args:
            scene_idx (int): Index of the scene to update
            status (str): New status for the scene
            assembled_video_path (Optional[str]): Path to the assembled video if available
        """
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
        """
        Update final video information.
        
        If status is "generated", marks the project as completed.
        
        Args:
            path (str): Path to the final video
            status (str): Status of the final video
            full_narration_text (str): Complete narration text
            hashtags (List[str]): List of hashtags for the video
        """
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
        """
        Get the next pending task in the generation pipeline.
        
        Tasks are returned in the following order:
        1. Script generation (if no script exists)
        2. Audio generation (for pending narration parts)
        3. Scene creation (for narration parts with audio)
        4. Chunk generation (for pending chunks)
        5. Scene assembly (for scenes with all chunks generated)
        6. Final assembly (if all scenes are completed)
        
        Returns:
            Tuple[str, Dict]: Task name and task data, or (None, None) if no pending tasks
        """
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
                if chunk["status"] != "generated":
                    return "generate_chunk", {
                        "scene_idx": scene["scene_idx"],
                        "chunk_idx": chunk["chunk_idx"],
                        "visual_prompt": chunk["visual_prompt"],
                        "motion_prompt": chunk.get("motion_prompt")
                    }
                    
            # Check if scene needs assembly
            if scene["status"] != "completed" and all(c["status"] == "video_generated" for c in scene["chunks"]):
                return "assemble_scene", {"scene_idx": scene["scene_idx"]}
                
        # Check if final video needs assembly
        if all(s["status"] == "completed" for s in self.state.scenes) and self.state.final_video["status"] != "generated":
            return "assemble_final", {}
            
        return None, None
        
    def get_scene_info(self, scene_idx: int) -> Optional[Dict]:
        """
        Get information about a specific scene.
        
        Args:
            scene_idx (int): Index of the scene to retrieve
            
        Returns:
            Optional[Dict]: Scene information if found, None otherwise
        """
        if not self.state:
            return None
            
        return next((s for s in self.state.scenes if s["scene_idx"] == scene_idx), None)
        
    def get_chunk_info(self, scene_idx: int, chunk_idx: int) -> Optional[Dict]:
        """
        Get information about a specific chunk.
        
        Args:
            scene_idx (int): Index of the scene containing the chunk
            chunk_idx (int): Index of the chunk to retrieve
            
        Returns:
            Optional[Dict]: Chunk information if found, None otherwise
        """
        scene = self.get_scene_info(scene_idx)
        if not scene:
            return None
            
        return next((c for c in scene["chunks"] if c["chunk_idx"] == chunk_idx), None)
        
    def is_completed(self) -> bool:
        """
        Check if the project is completed.
        
        Returns:
            bool: True if project status is "completed", False otherwise
        """
        if not self.state:
            return False
            
        return self.state.project_info["status"] == "completed"
        
    def update_chunk_content(self, scene_idx: int, chunk_idx: int, 
                           visual_prompt: Optional[str] = None,
                           motion_prompt: Optional[str] = None) -> None:
        """
        Update chunk content and trigger dependency chain if needed.
        
        If the content changes, clears generated assets and triggers regeneration
        of affected components.
        
        Args:
            scene_idx (int): Index of the scene containing the chunk
            chunk_idx (int): Index of the chunk to update
            visual_prompt (Optional[str]): New visual prompt if provided
            motion_prompt (Optional[str]): New motion prompt if provided
        """
        if not self.state:
            return
            
        scene = next((s for s in self.state.scenes if s["scene_idx"] == scene_idx), None)
        if not scene:
            return
            
        chunk = next((c for c in scene["chunks"] if c["chunk_idx"] == chunk_idx), None)
        if not chunk:
            return
            
        # Check if content is actually changing
        content_changed = False
        if visual_prompt and visual_prompt != chunk["visual_prompt"]:
            chunk["visual_prompt"] = visual_prompt
            content_changed = True
        if motion_prompt and motion_prompt != chunk.get("motion_prompt"):
            chunk["motion_prompt"] = motion_prompt
            content_changed = True
            
        # If content changed, mark for regeneration
        if content_changed:
            # Clear generated assets
            chunk["keyframe_image_path"] = ""
            chunk["video_path"] = ""
            # Mark chunk for regeneration
            chunk["status"] = "pending"
            # Trigger dependency chain
            self._mark_scene_for_reassembly(scene_idx)
            self._mark_final_for_reassembly()
            
        self._save_state()
        
    def update_scene_narration(self, scene_idx: int, text: Optional[str] = None) -> None:
        """
        Update scene narration and trigger dependency chain if needed.
        
        If the narration text changes, clears audio assets and triggers regeneration
        of affected components.
        
        Args:
            scene_idx (int): Index of the scene to update
            text (Optional[str]): New narration text if provided
        """
        if not self.state:
            return
            
        scene = next((s for s in self.state.scenes if s["scene_idx"] == scene_idx), None)
        if not scene:
            return
            
        # Update narration text if provided
        if text and text != scene["narration"]["text"]:
            scene["narration"]["text"] = text
            scene["narration"]["status"] = "pending"
            scene["narration"]["audio_path"] = ""
            scene["narration"]["duration"] = 0
            # Trigger dependency chain
            self._mark_scene_for_reassembly(scene_idx)
            self._mark_final_for_reassembly()
            
        self._save_state() 