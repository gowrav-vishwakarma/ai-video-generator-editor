import streamlit as st
from task_executor import TaskExecutor
from config_manager import ContentConfig, ModuleSelectorConfig
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)

class UITaskExecutor:
    """Handles task execution triggered from the Streamlit UI, providing user feedback."""
    
    def __init__(self, project_manager):
        self.project_manager = project_manager
        self.task_executor: Optional[TaskExecutor] = None
        self._initialize_task_executor()
        
    def _initialize_task_executor(self):
        """Initializes the core task executor with project configuration."""
        if not self.project_manager.state:
            st.error("Cannot initialize task executor: Project state not found.")
            return
            
        try:
            content_cfg = ContentConfig(**self.project_manager.state.project_info["config"])
            module_selector_cfg = ModuleSelectorConfig()
            self.task_executor = TaskExecutor(self.project_manager, content_cfg, module_selector_cfg)
        except Exception as e:
            logger.error(f"Failed to initialize TaskExecutor: {e}")
            st.error(f"Configuration Error: {e}")

    def update_narration_text(self, scene_idx: int, text: str):
        """Saves edited narration text via the project manager."""
        self.project_manager.update_narration_part_text(scene_idx, text)

    def update_chunk_prompts(self, scene_idx: int, chunk_idx: int, visual_prompt: Optional[str] = None, motion_prompt: Optional[str] = None):
        """Saves edited chunk prompts via the project manager."""
        self.project_manager.update_chunk_content(scene_idx, chunk_idx, visual_prompt, motion_prompt)

    def regenerate_audio(self, scene_idx: int, text: str, speaker_audio: Optional[str] = None) -> bool:
        """Generates or regenerates audio for a specific script part."""
        if not self.task_executor:
            self._initialize_task_executor()
            if not self.task_executor: return False
        
        # Ensure the text is saved and dependencies are reset before generating
        self.project_manager.update_narration_part_text(scene_idx, text)
        
        task_data = {
            "scene_idx": scene_idx, 
            "text": text,
            "speaker_wav": speaker_audio if speaker_audio and os.path.exists(speaker_audio) else None
        }
        
        success = self.task_executor.execute_task("generate_audio", task_data)
        if success:
            st.toast(f"Audio for part {scene_idx + 1} generated!", icon="🔊")
        else:
            st.error(f"Failed to generate audio for part {scene_idx + 1}.")
        return success
            
    def create_scene(self, scene_idx: int) -> bool:
        """Creates a scene (with chunks) from a narration part."""
        if not self.task_executor: return False
        success = self.task_executor.execute_task("create_scene", {"scene_idx": scene_idx})
        if success:
            st.toast(f"Scene for part {scene_idx + 1} created!", icon="🎬")
        else:
            st.error(f"Failed to create scene for part {scene_idx + 1}.")
        return success

    def regenerate_chunk(self, scene_idx: int, chunk_idx: int) -> bool:
        """Generates or regenerates a specific video chunk."""
        if not self.task_executor: return False
        chunk = self.project_manager.get_scene_info(scene_idx)['chunks'][chunk_idx]
        
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
            st.toast(f"Video chunk {chunk_idx + 1} for scene {scene_idx + 1} generated!", icon="📹")
        else:
            st.error(f"Failed to generate chunk.")
        return success
            
    def assemble_final_video(self) -> bool:
        """Assembles the final video from all completed scenes."""
        if not self.task_executor: return False
        success = self.task_executor.execute_task("assemble_final", {})
        if success:
            st.toast("Final video assembled successfully!", icon="🏆")
        else:
            st.error("Failed to assemble final video.")
        return success