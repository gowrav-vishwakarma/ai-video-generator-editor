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
            # When creating ContentConfig, we need to handle the case where it might have been
            # saved with the old property-based structure.
            config_dict = self.project_manager.state.project_info["config"]
            if '_generation_resolution' in config_dict:
                config_dict['generation_resolution'] = config_dict.pop('_generation_resolution')
            
            content_cfg = ContentConfig(**config_dict)
            module_selector_cfg = ModuleSelectorConfig()
            self.task_executor = TaskExecutor(self.project_manager, content_cfg, module_selector_cfg)
        except Exception as e:
            logger.error(f"Failed to initialize TaskExecutor: {e}")
            st.error(f"Configuration Error: {e}")

    def update_narration_text(self, scene_idx: int, text: str):
        self.project_manager.update_narration_part_text(scene_idx, text)

    def update_chunk_prompts(self, scene_idx: int, chunk_idx: int, visual_prompt: Optional[str] = None, motion_prompt: Optional[str] = None):
        self.project_manager.update_chunk_content(scene_idx, chunk_idx, visual_prompt, motion_prompt)

    def regenerate_audio(self, scene_idx: int, text: str, speaker_audio: Optional[str] = None) -> bool:
        if not self.task_executor: return False
        self.project_manager.update_narration_part_text(scene_idx, text)
        task_data = {
            "scene_idx": scene_idx, 
            "text": text,
            "speaker_wav": speaker_audio if speaker_audio and os.path.exists(speaker_audio) else None
        }
        success = self.task_executor.execute_task("generate_audio", task_data)
        if success:
            st.toast(f"Audio for Scene {scene_idx + 1} generated!", icon="🔊")
        else:
            st.error(f"Failed to generate audio for Scene {scene_idx + 1}.")
        self.project_manager.load_project() # Reload state
        return success
            
    def create_scene(self, scene_idx: int) -> bool:
        if not self.task_executor: return False
        success = self.task_executor.execute_task("create_scene", {"scene_idx": scene_idx})
        if success:
            st.toast(f"Scene {scene_idx + 1} created!", icon="🎬")
        else:
            st.error(f"Failed to create Scene {scene_idx + 1}.")
        self.project_manager.load_project() # Reload state
        return success

    # #############################################################################
    # # --- THIS IS THE MISSING CODE YOU NEED ---
    # #############################################################################

    def regenerate_chunk_image(self, scene_idx: int, chunk_idx: int) -> bool:
        """Generates or regenerates ONLY the keyframe image for a specific chunk."""
        if not self.task_executor: return False
        
        # In manual mode, we must first reset the status of this chunk and its dependencies
        self.project_manager.update_chunk_content(scene_idx, chunk_idx, visual_prompt=self.project_manager.get_scene_info(scene_idx)['chunks'][chunk_idx]['visual_prompt'])

        chunk = self.project_manager.get_scene_info(scene_idx)['chunks'][chunk_idx]
        image_task_data = {
            "scene_idx": scene_idx,
            "chunk_idx": chunk_idx,
            "visual_prompt": chunk["visual_prompt"],
        }
        
        success = self.task_executor.execute_task("generate_chunk_image", image_task_data)
        
        if success:
            st.toast(f"Image for Chunk {chunk_idx + 1} generated!", icon="🖼️")
        else:
            st.error(f"Failed to generate image for Chunk {chunk_idx + 1}.")
            
        self.project_manager.load_project() # Reload to get the new image path
        return success

    def regenerate_chunk_video(self, scene_idx: int, chunk_idx: int) -> bool:
        """Generates or regenerates ONLY the video for a specific chunk, using the existing image."""
        if not self.task_executor: return False

        chunk = self.project_manager.get_scene_info(scene_idx)['chunks'][chunk_idx]
        video_task_data = {
            "scene_idx": scene_idx,
            "chunk_idx": chunk_idx,
            "motion_prompt": chunk.get("motion_prompt")
        }

        success = self.task_executor.execute_task("generate_chunk_video", video_task_data)

        if success:
            st.toast(f"Video for Chunk {chunk_idx + 1} generated!", icon="📹")
        else:
            st.error(f"Failed to generate video for Chunk {chunk_idx + 1}.")

        self.project_manager.load_project() # Reload to get the new video path
        return success
    # #############################################################################
    # # --- END OF MISSING CODE ---
    # #############################################################################
            
    def assemble_final_video(self) -> bool:
        if not self.task_executor: return False
        success = self.task_executor.execute_task("assemble_final", {})
        if success:
            st.toast("Final video assembled successfully!", icon="🏆")
        else:
            st.error("Failed to assemble final video.")
        return success