import streamlit as st
from task_executor import TaskExecutor
from config_manager import ContentConfig, ModuleSelectorConfig
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)

class UITaskExecutor:
    """Handles task execution in the UI context"""
    
    def __init__(self, project_manager):
        self.project_manager = project_manager
        self.task_executor = None
        self._initialize_task_executor()
        
    def _initialize_task_executor(self):
        """Initialize the task executor with project configuration"""
        if not self.project_manager.state:
            return
            
        content_cfg = ContentConfig(**self.project_manager.state.project_info["config"])
        module_selector_cfg = ModuleSelectorConfig()
        self.task_executor = TaskExecutor(self.project_manager, content_cfg, module_selector_cfg)
        
    def regenerate_audio(self, scene_idx: int, text: str, speaker_audio: Optional[str] = None) -> bool:
        """Regenerate audio for a specific scene"""
        try:
            if not self.task_executor:
                self._initialize_task_executor()
            
            logger.info(f"Regenerating audio for scene {scene_idx}")
            logger.info(f"Speaker audio path: {speaker_audio}")
            logger.info(f"Speaker audio exists: {os.path.exists(speaker_audio) if speaker_audio else False}")
            
            if not speaker_audio or not os.path.exists(speaker_audio):
                st.warning("No valid speaker audio file provided. Using default voice.")
                
            # Update the narration text
            self.project_manager.update_scene_narration(scene_idx, text)
            
            # Execute audio generation
            task_data = {
                "scene_idx": scene_idx, 
                "text": text,
                "speaker_wav": speaker_audio if speaker_audio and os.path.exists(speaker_audio) else None
            }
            logger.info(f"Task data: {task_data}")
            
            success = self.task_executor.execute_task("generate_audio", task_data)
            
            if success:
                st.success(f"Audio regenerated for scene {scene_idx + 1}")
            else:
                st.error(f"Failed to regenerate audio for scene {scene_idx + 1}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error regenerating audio: {e}")
            st.error(f"Error regenerating audio: {str(e)}")
            return False
            
    def regenerate_chunk(self, scene_idx: int, chunk_idx: int) -> bool:
        """Regenerate a specific chunk"""
        try:
            if not self.task_executor:
                self._initialize_task_executor()
                
            # Get chunk info
            chunk = self.project_manager.get_chunk_info(scene_idx, chunk_idx)
            if not chunk:
                st.error(f"Chunk not found: Scene {scene_idx + 1}, Chunk {chunk_idx + 1}")
                return False
                
            # Execute chunk generation
            success = self.task_executor.execute_task(
                "generate_chunk",
                {
                    "scene_idx": scene_idx,
                    "chunk_idx": chunk_idx,
                    "visual_prompt": chunk["visual_prompt"],
                    "motion_prompt": chunk.get("motion_prompt")
                }
            )
            
            if success:
                st.success(f"Chunk regenerated: Scene {scene_idx + 1}, Chunk {chunk_idx + 1}")
            else:
                st.error(f"Failed to regenerate chunk: Scene {scene_idx + 1}, Chunk {chunk_idx + 1}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error regenerating chunk: {e}")
            st.error(f"Error regenerating chunk: {str(e)}")
            return False
            
    def assemble_final_video(self) -> bool:
        """Assemble the final video"""
        try:
            if not self.task_executor:
                self._initialize_task_executor()
                
            # Execute final assembly
            success = self.task_executor.execute_task("assemble_final", {})
            
            if success:
                st.success("Final video assembled successfully")
            else:
                st.error("Failed to assemble final video")
                
            return success
            
        except Exception as e:
            logger.error(f"Error assembling final video: {e}")
            st.error(f"Error assembling final video: {str(e)}")
            return False
            
    def add_new_script_part(self, text: str) -> bool:
        """Add a new script part"""
        try:
            if not self.project_manager.state:
                return False
                
            # Add new narration part
            new_part = {
                "text": text,
                "status": "pending",
                "audio_path": "",
                "duration": 0
            }
            
            self.project_manager.state.script["narration_parts"].append(new_part)
            self.project_manager._save_state()
            
            st.success("New script part added")
            return True
            
        except Exception as e:
            logger.error(f"Error adding new script part: {e}")
            st.error(f"Error adding new script part: {str(e)}")
            return False

    def create_scene(self, scene_idx: int) -> bool:
        """Create a new scene for a narration part"""
        try:
            if not self.task_executor:
                self._initialize_task_executor()
            
            logger.info(f"Creating scene for narration part {scene_idx}")
            success = self.task_executor.execute_task(
                "create_scene",
                {"scene_idx": scene_idx}
            )
            
            if success:
                st.success(f"Scene created for part {scene_idx + 1}")
            else:
                st.error(f"Failed to create scene for part {scene_idx + 1}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error creating scene: {e}")
            st.error(f"Error creating scene: {str(e)}")
            return False
            
    def generate_chunk(self, scene_idx: int, chunk_idx: int, visual_prompt: str, motion_prompt: Optional[str] = None) -> bool:
        """Generate a video chunk for a scene"""
        try:
            if not self.task_executor:
                self._initialize_task_executor()
            
            logger.info(f"Generating chunk {chunk_idx} for scene {scene_idx}")
            success = self.task_executor.execute_task(
                "generate_chunk",
                {
                    "scene_idx": scene_idx,
                    "chunk_idx": chunk_idx,
                    "visual_prompt": visual_prompt,
                    "motion_prompt": motion_prompt
                }
            )
            
            if success:
                st.success(f"Chunk {chunk_idx + 1} generated for scene {scene_idx + 1}")
            else:
                st.error(f"Failed to generate chunk {chunk_idx + 1} for scene {scene_idx + 1}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error generating chunk: {e}")
            st.error(f"Error generating chunk: {str(e)}")
            return False 