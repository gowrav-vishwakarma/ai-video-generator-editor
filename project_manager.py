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

Status Definitions:
- "pending": Initial state, needs to be processed
- "in_progress": Currently being processed
- "completed": Successfully processed and ready for next stage
- "failed": Processing failed, needs retry
- "generated": Final state for chunks and final video
- "image_generated": Intermediate state for chunks (keyframe generated)
- "video_generated": Final state for chunks (video generated)

Valid Status Transitions:
1. Script Parts: pending -> generated
2. Visual Prompts: pending -> generated
3. Scenes: pending -> in_progress -> completed
4. Chunks: pending -> image_generated -> video_generated
5. Final Video: pending -> generated

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
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from config_manager import ContentConfig

# Configure logging
logger = logging.getLogger(__name__)

# Status validation
VALID_STATUSES = {
    "script": ["pending", "generated"],
    "visual_prompt": ["pending", "generated"],
    "scene": ["pending", "in_progress", "completed", "failed"],
    "chunk": ["pending", "image_generated", "video_generated", "failed"],
    "final_video": ["pending", "generated", "failed"]
}

# Define final statuses for each component type
FINAL_STATUSES = {
    "script": "generated",
    "visual_prompt": "generated",
    "scene": "completed",
    "chunk": "video_generated",
    "final_video": "generated"
}

VALID_STATUS_TRANSITIONS = {
    "script": {"pending": ["generated"]},
    "visual_prompt": {"pending": ["generated"]},
    "scene": {
        "pending": ["in_progress", "failed"],
        "in_progress": ["completed", "failed"],
        "completed": ["pending", "failed"],
        "failed": ["pending"]
    },
    "chunk": {
        "pending": ["image_generated", "failed"],
        "image_generated": ["video_generated", "failed"],
        "video_generated": ["pending", "failed"],
        "failed": ["pending"]
    },
    "final_video": {
        "pending": ["generated", "failed"],
        "generated": ["pending", "failed"],
        "failed": ["pending"]
    }
}

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
        
    def _is_final_status(self, component_type: str, status: str) -> bool:
        """
        Check if a status is the final status for a component type.
        
        Args:
            component_type (str): Type of component
            status (str): Status to check
            
        Returns:
            bool: True if status is final, False otherwise
        """
        return status == FINAL_STATUSES.get(component_type)
        
    def _validate_status_transition(self, component_type: str, old_status: str, new_status: str) -> bool:
        """
        Validate if a status transition is allowed.
        
        Args:
            component_type (str): Type of component (script, visual_prompt, scene, chunk, final_video)
            old_status (str): Current status
            new_status (str): New status to transition to
            
        Returns:
            bool: True if transition is valid, False otherwise
        """
        # If new status is final, always allow it
        if self._is_final_status(component_type, new_status):
            return True
            
        # If old status is final and new status is not, always allow it (regeneration)
        if self._is_final_status(component_type, old_status):
            return True
            
        # For non-final statuses, validate against transitions
        if component_type not in VALID_STATUS_TRANSITIONS:
            logger.error(f"Invalid component type: {component_type}")
            return False
            
        if old_status not in VALID_STATUS_TRANSITIONS[component_type]:
            logger.error(f"Invalid old status {old_status} for {component_type}")
            return False
            
        if new_status not in VALID_STATUS_TRANSITIONS[component_type][old_status]:
            logger.error(f"Invalid status transition from {old_status} to {new_status} for {component_type}")
            return False
            
        return True
        
    def _log_status_change(self, component_type: str, component_id: str, old_status: str, new_status: str) -> None:
        """
        Log a status change with validation.
        
        Args:
            component_type (str): Type of component
            component_id (str): Identifier for the component
            old_status (str): Current status
            new_status (str): New status
        """
        if self._validate_status_transition(component_type, old_status, new_status):
            logger.info(f"{component_type} {component_id}: {old_status} -> {new_status}")
        else:
            logger.error(f"Invalid status transition attempted: {component_type} {component_id}: {old_status} -> {new_status}")
            
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
        logger.info(f"Initialized new project: {topic}")
        
    def load_project(self) -> bool:
        """
        Load project state from project.json.
        
        Validates all statuses in the project to ensure they are valid according to
        the defined status rules. If any invalid status is found, the project loading
        fails immediately.
        
        Returns:
            bool: True if project was successfully loaded, False otherwise
        """
        if not os.path.exists(self.project_file):
            logger.warning(f"Project file not found: {self.project_file}")
            return False
            
        try:
            with open(self.project_file, 'r') as f:
                data = json.load(f)
                
            # Create state first
            self.state = ProjectState(**data)
            
            # Check for any non-final statuses and trigger dependency chain
            # Check script parts
            for i, part in enumerate(self.state.script["narration_parts"]):
                if not self._is_final_status("script", part.get("status", "")):
                    part["status"] = "pending"
                    
            # Check visual prompts
            for i, prompt in enumerate(self.state.script["visual_prompts"]):
                if not self._is_final_status("visual_prompt", prompt.get("status", "")):
                    prompt["status"] = "pending"
                    
            # Check scenes and their chunks
            for scene in self.state.scenes:
                # Check scene status
                if not self._is_final_status("scene", scene.get("status", "")):
                    scene["status"] = "pending"
                    scene["assembled_video_path"] = ""
                    
                # Check chunks
                for chunk in scene["chunks"]:
                    if not self._is_final_status("chunk", chunk.get("status", "")):
                        chunk["status"] = "pending"
                        chunk["keyframe_image_path"] = ""
                        chunk["video_path"] = ""
                        
            # Check final video status
            if not self._is_final_status("final_video", self.state.final_video.get("status", "")):
                self.state.final_video["status"] = "pending"
                self.state.final_video["path"] = ""
                
            # Save the updated state
            self._save_state()
            logger.info(f"Loaded project: {self.state.project_info['topic']}")
            return True
        except Exception as e:
            logger.error(f"Error loading project: {e}")
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
        logger.info("Updated script with new narration parts and visual prompts")
        
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
            old_status = self.state.scenes[existing_scene_idx]["status"]
            self._log_status_change("scene", f"Scene {scene_idx+1}", old_status, "pending")
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
            logger.info(f"Added new scene {scene_idx+1}")
            
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
            old_status = scene["status"]
            self._log_status_change("scene", f"Scene {scene_idx+1}", old_status, "pending")
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
            
        old_status = self.state.final_video["status"]
        self._log_status_change("final_video", "Final Video", old_status, "pending")
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
            
        # If chunk status is changing from video_generated to something else
        if chunk["status"] == "video_generated" and status != "video_generated":
            # Mark scene for reassembly
            self._mark_scene_for_reassembly(scene_idx)
            # Mark final video for reassembly
            self._mark_final_for_reassembly()
            
        old_status = chunk["status"]
        self._log_status_change("chunk", f"Scene {scene_idx+1} Chunk {chunk_idx+1}", old_status, status)
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
            
        # If scene status is changing from completed to something else
        if scene["status"] == "completed" and status != "completed":
            # Mark final video for reassembly
            self._mark_final_for_reassembly()
            
        old_status = scene["status"]
        self._log_status_change("scene", f"Scene {scene_idx+1}", old_status, status)
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
            
        old_status = self.state.final_video["status"]
        self._log_status_change("final_video", "Final Video", old_status, status)
        
        self.state.final_video.update({
            "path": path,
            "status": status,
            "full_narration_text": full_narration_text,
            "hashtags": hashtags
        })
        
        if status == "generated":
            self.state.project_info["status"] = "completed"
            logger.info("Project marked as completed")
            
        self._save_state()
        
    def get_next_pending_task(self) -> Tuple[str, Dict]:
        """
        Get the next pending task in the generation pipeline.
        
        Tasks are returned in the following order:
        1. Script generation (if no script exists)
        2. Audio generation (for non-final narration parts)
        3. Scene creation (for narration parts with audio)
        4. Chunk generation (for non-final chunks)
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
            if not self._is_final_status("script", narration.get("status", "")):
                return "generate_audio", {"scene_idx": i, "text": narration["text"]}
                
        # Check scene creation for each narration part
        for i, narration in enumerate(self.state.script["narration_parts"]):
            # Find if scene exists for this narration
            scene_exists = any(s["scene_idx"] == i for s in self.state.scenes)
            if self._is_final_status("script", narration.get("status", "")) and narration.get("audio_path") and not scene_exists:
                return "create_scene", {"scene_idx": i}
                
        # Check video generation for each scene
        for scene in self.state.scenes:
            # Check chunks that need generation
            for chunk in scene["chunks"]:
                if not self._is_final_status("chunk", chunk.get("status", "")):
                    return "generate_chunk", {
                        "scene_idx": scene["scene_idx"],
                        "chunk_idx": chunk["chunk_idx"],
                        "visual_prompt": chunk["visual_prompt"],
                        "motion_prompt": chunk.get("motion_prompt")
                    }
                    
            # Check if scene needs assembly
            if not self._is_final_status("scene", scene.get("status", "")) and all(self._is_final_status("chunk", c.get("status", "")) for c in scene["chunks"]):
                return "assemble_scene", {"scene_idx": scene["scene_idx"]}
                
        # Check if final video needs assembly
        if all(self._is_final_status("scene", s.get("status", "")) for s in self.state.scenes) and not self._is_final_status("final_video", self.state.final_video.get("status", "")):
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
            old_status = chunk["status"]
            self._log_status_change("chunk", f"Scene {scene_idx+1} Chunk {chunk_idx+1}", old_status, "pending")
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
            old_status = scene["narration"].get("status", "pending")
            self._log_status_change("script", f"Scene {scene_idx+1} Narration", old_status, "pending")
            scene["narration"]["status"] = "pending"
            scene["narration"]["audio_path"] = ""
            scene["narration"]["duration"] = 0
            # Trigger dependency chain
            self._mark_scene_for_reassembly(scene_idx)
            self._mark_final_for_reassembly()
            
        self._save_state()
        
    def mark_tasks_pending_after(self, task: str) -> None:
        """
        Mark all tasks after the specified task as pending.
        This is used when resuming from a specific point in the pipeline.
        
        Args:
            task (str): Task to start from (all tasks after this will be marked pending)
        """
        if not self.state:
            return
            
        # Define task order
        task_order = [
            "generate_script",
            "generate_audio",
            "create_scene",
            "generate_chunk",
            "assemble_scene",
            "assemble_final"
        ]
        
        # Find the index of the specified task
        try:
            start_idx = task_order.index(task)
        except ValueError:
            logger.error(f"Unknown task: {task}")
            return
            
        # Mark tasks as pending based on their order
        if start_idx <= 0:  # generate_script
            # Reset all script parts
            for part in self.state.script["narration_parts"]:
                part["status"] = "pending"
                part["audio_path"] = ""
                part["duration"] = 0
            for prompt in self.state.script["visual_prompts"]:
                prompt["status"] = "pending"
                
        if start_idx <= 1:  # generate_audio
            # Reset all audio generation
            for part in self.state.script["narration_parts"]:
                part["status"] = "pending"
                part["audio_path"] = ""
                part["duration"] = 0
                
        if start_idx <= 2:  # create_scene
            # Reset all scenes
            self.state.scenes = []
            
        if start_idx <= 3:  # generate_chunk
            # Reset all chunks in existing scenes
            for scene in self.state.scenes:
                for chunk in scene["chunks"]:
                    chunk["status"] = "pending"
                    chunk["keyframe_image_path"] = ""
                    chunk["video_path"] = ""
                scene["status"] = "pending"
                scene["assembled_video_path"] = ""
                
        if start_idx <= 4:  # assemble_scene
            # Reset all scene assemblies
            for scene in self.state.scenes:
                scene["status"] = "pending"
                scene["assembled_video_path"] = ""
                
        if start_idx <= 5:  # assemble_final
            # Reset final video
            self.state.final_video.update({
                "path": "",
                "status": "pending",
                "full_narration_text": "",
                "hashtags": []
            })
            
        self._save_state()
        logger.info(f"Marked all tasks after {task} as pending") 