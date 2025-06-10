# In ui_task_executor.py

import streamlit as st
from task_executor import TaskExecutor
from config_manager import ContentConfig
import logging
from typing import List, Optional
import os
from utils import load_and_correct_image_orientation

logger = logging.getLogger(__name__)

class UITaskExecutor:
    """Handles task execution triggered from the Streamlit UI, providing user feedback."""
    
    def __init__(self, project_manager):
        self.project_manager = project_manager
        self.task_executor: Optional[TaskExecutor] = None
        self._initialize_task_executor()
        
    def _initialize_task_executor(self):
        if not self.project_manager.state:
            st.error("Cannot initialize task executor: Project state not found.")
            return
        try:
            # TaskExecutor now only needs the project_manager to initialize itself
            self.task_executor = TaskExecutor(self.project_manager)
        except Exception as e:
            logger.error(f"Failed to initialize TaskExecutor: {e}", exc_info=True)
            st.error(f"Configuration Error: {e}")

    def update_narration_text(self, scene_idx: int, text: str):
        self.project_manager.update_narration_part_text(scene_idx, text)

    def update_chunk_prompts(self, scene_idx: int, chunk_idx: int, visual_prompt: Optional[str] = None, motion_prompt: Optional[str] = None):
        self.project_manager.update_chunk_content(scene_idx, chunk_idx, visual_prompt, motion_prompt)

    def regenerate_audio(self, scene_idx: int, text: str, speaker_audio: Optional[str] = None) -> bool:
        if not self.task_executor: return False
        self.project_manager.update_narration_part_text(scene_idx, text)
        task_data = {"scene_idx": scene_idx, "text": text, "speaker_wav": speaker_audio if speaker_audio and os.path.exists(speaker_audio) else None}
        success = self.task_executor.execute_task("generate_audio", task_data)
        if success: st.toast(f"Audio for Scene {scene_idx + 1} generated!", icon="🔊")
        else: st.error(f"Failed to generate audio for Scene {scene_idx + 1}.")
        self.project_manager.load_project()
        return success
            
    def create_scene(self, scene_idx: int) -> bool:
        if not self.task_executor: return False
        success = self.task_executor.execute_task("create_scene", {"scene_idx": scene_idx})
        if success: st.toast(f"Scene {scene_idx + 1} created!", icon="🎬")
        else: st.error(f"Failed to create Scene {scene_idx + 1}.")
        self.project_manager.load_project()
        return success

    def regenerate_chunk_image(self, scene_idx: int, chunk_idx: int) -> bool:
        if not self.task_executor: return False
        # No need to call update_chunk_content if prompts are not changing
        # It resets status, which is what we want for a regen.
        self.project_manager.update_chunk_content(scene_idx, chunk_idx) 
        chunk = self.project_manager.get_scene_info(scene_idx).chunks[chunk_idx]
        task_data = {"scene_idx": scene_idx, "chunk_idx": chunk_idx, "visual_prompt": chunk.visual_prompt}
        success = self.task_executor.execute_task("generate_chunk_image", task_data)
        if success: st.toast(f"Image for Chunk {chunk_idx + 1} generated!", icon="🖼️")
        else: st.error(f"Failed to generate image for Chunk {chunk_idx + 1}.")
        self.project_manager.load_project()
        return success

    def regenerate_chunk_video(self, scene_idx: int, chunk_idx: int) -> bool:
        if not self.task_executor: return False
        # Also reset status for regen
        self.project_manager.update_chunk_content(scene_idx, chunk_idx)
        chunk = self.project_manager.get_scene_info(scene_idx).chunks[chunk_idx]
        task_data = {
            "scene_idx": scene_idx, "chunk_idx": chunk_idx,
            "visual_prompt": chunk.visual_prompt,
            "motion_prompt": chunk.motion_prompt
        }
        success = self.task_executor.execute_task("generate_chunk_video", task_data)
        if success: st.toast(f"Video for Chunk {chunk_idx + 1} generated!", icon="📹")
        else: st.error(f"Failed to generate video for Chunk {chunk_idx + 1}.")
        self.project_manager.load_project()
        return success

    def regenerate_chunk_t2v(self, scene_idx: int, chunk_idx: int) -> bool:
        if not self.task_executor: return False
        self.project_manager.update_chunk_content(scene_idx, chunk_idx)
        chunk = self.project_manager.get_scene_info(scene_idx).chunks[chunk_idx]
        task_data = {"scene_idx": scene_idx, "chunk_idx": chunk_idx, "visual_prompt": chunk.visual_prompt}
        success = self.task_executor.execute_task("generate_chunk_t2v", task_data)
        if success: st.toast(f"T2V Chunk {chunk_idx + 1} generated!", icon="📹")
        else: st.error(f"Failed to generate T2V Chunk {chunk_idx + 1}.")
        self.project_manager.load_project()
        return success
            
    def assemble_final_video(self) -> bool:
        if not self.task_executor: return False
        success = self.task_executor.execute_task("assemble_final", {})
        if success: st.toast("Final video assembled successfully!", icon="🏆")
        else: st.error("Failed to assemble final video.")
        self.project_manager.load_project()
        return success

    def add_character(self, name: str, image_file: "UploadedFile"):
        if not self.project_manager.state: return False
        safe_name = name.replace(" ", "_")
        char_dir = os.path.join(self.project_manager.output_dir, "characters", safe_name)
        os.makedirs(char_dir, exist_ok=True)
        ref_image_path = os.path.join(char_dir, "reference.png")
        
        # --- FIX: Correct orientation before saving ---
        corrected_image = load_and_correct_image_orientation(image_file)
        if corrected_image:
            corrected_image.save(ref_image_path, "PNG")
            char_data = {"name": name, "reference_image_path": ref_image_path}
            self.project_manager.add_character(char_data)
            st.toast(f"Character '{name}' added!", icon="👤")
            return True
        else:
            st.error(f"Could not process image for new character {name}. Aborting.")
            return False

    def update_character(self, old_name: str, new_name: str, new_image_file: Optional["UploadedFile"]):
        ref_image_path = None
        if new_image_file:
            safe_name = (new_name or old_name).replace(" ", "_")
            char_dir = os.path.join(self.project_manager.output_dir, "characters", safe_name)
            os.makedirs(char_dir, exist_ok=True)
            ref_image_path = os.path.join(char_dir, "reference.png")

            corrected_image = load_and_correct_image_orientation(new_image_file)
            if corrected_image:
                corrected_image.save(ref_image_path, "PNG")
            else:
                # If image processing fails, don't update the image path.
                st.error("Failed to process the new image. Character image was not updated.")
                ref_image_path = None 

        self.project_manager.update_character(old_name, new_name, ref_image_path)
        st.toast(f"Character '{old_name}' updated!", icon="✏️")
        return True

    def delete_character(self, name: str):
        self.project_manager.delete_character(name)
        st.toast(f"Character '{name}' deleted!", icon="🗑️")
        return True
    
    def update_scene_characters(self, scene_idx: int, character_names: List[str]):
        self.project_manager.update_scene_characters(scene_idx, character_names)
        st.toast(f"Characters for Scene {scene_idx+1} updated.", icon="🎬")
    
    # NEW METHOD: Add Scene
    def add_new_scene(self, scene_idx: int):
        """UI wrapper to add a new scene."""
        self.project_manager.add_new_scene_at(scene_idx)
        st.toast(f"New scene added at position {scene_idx + 1}!", icon="➕")
        return True

    # NEW METHOD: Remove Scene
    def remove_scene(self, scene_idx: int):
        """UI wrapper to remove a scene."""
        self.project_manager.remove_scene_at(scene_idx)
        st.toast(f"Scene {scene_idx + 1} removed!", icon="🗑️")
        return True