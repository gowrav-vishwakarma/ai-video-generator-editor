# In ui_task_executor.py
import streamlit as st
import os
from uuid import UUID
from project_manager import ProjectManager
from task_executor import TaskExecutor
from utils import load_and_correct_image_orientation
from config_manager import Shot
from importlib import import_module
from system import select_item

def _import_class(module_path_str: str):
    if not module_path_str: return None
    try:
        module_path, class_name = module_path_str.rsplit('.', 1); module = import_module(module_path); return getattr(module, class_name)
    except Exception as e:
        print(f"Warning: Could not import module: {module_path_str}. Error: {e}"); return None

class UITaskExecutor:
    def __init__(self, project_manager: ProjectManager):
        self.pm = project_manager; self._task_executor = None
    @property
    def task_executor(self) -> TaskExecutor:
        if self._task_executor is None: self._task_executor = TaskExecutor(self.pm)
        return self._task_executor
    
    def get_shot_duration(self, shot: Shot) -> float:
        scene = next((s for s in self.pm.state.scenes if shot in s.shots), None)
        if not scene or not scene.narration.duration > 0 or not scene.shots: return 0.0
        target_shot_duration = scene.narration.duration / len(scene.shots); flow = shot.generation_flow
        if flow in ("T2I_I2V", "Upload_I2V"): module_path = shot.module_selections.get('i2v')
        elif flow == "T2V": module_path = shot.module_selections.get('t2v')
        else: return 0.0
        if not module_path: return 0.0
        ModuleClass = _import_class(module_path)
        if not ModuleClass: return 0.0
        try:
            model_caps = ModuleClass.get_model_capabilities(); module_max_duration = model_caps.get("max_shot_duration", 0.0)
        except Exception: module_max_duration = 0.0
        return min(target_shot_duration, module_max_duration)

    def add_scene(self, after_scene_uuid: UUID = None):
        if not self.pm or not self.pm.state: return
        new_scene_uuid = self.pm.add_scene(after_scene_uuid)
        select_item('scene', new_scene_uuid)
        st.rerun()

    def update_scene_title(self, scene_uuid: UUID):
        key = f"title_{scene.uuid}"; new_title = st.session_state.get(key)
        if new_title: self.pm.update_scene(scene_uuid, {"title": new_title})
        
    def update_scene_narration(self, scene_uuid: UUID):
        key = f"narration_{scene.uuid}"; new_text = st.session_state.get(key)
        if new_text is not None: self.pm.update_scene(scene_uuid, {"narration_text": new_text})

    def update_scene_voice(self, scene_uuid: UUID):
        voice_key = f"voice_select_{scene_uuid}"; selected_voice_name = st.session_state.get(voice_key)
        voice_options = {v.name: v.uuid for v in self.pm.state.voices}
        if selected_voice_name and selected_voice_name in voice_options:
            new_voice_uuid = voice_options[selected_voice_name]
            scene = self.pm.get_scene(scene_uuid)
            if scene and scene.narration.voice_uuid != new_voice_uuid:
                self.pm.update_scene(scene_uuid, {"voice_uuid": new_voice_uuid}); st.toast("Voice updated.", icon="🗣️")
    
    def _execute_and_reload(self, task, data, success_msg, error_msg):
        with st.spinner(f"{error_msg.replace('failed', 'in progress')}..."): success = self.task_executor.execute_task(task, data)
        if success: st.toast(success_msg, icon="✨")
        else: st.error(error_msg)
        st.rerun()
        
    def create_character(self, name: str, image_file):
        char_dir = os.path.join(self.pm.output_dir, "characters", name.replace(" ", "_")); os.makedirs(char_dir, exist_ok=True)
        img_path = os.path.join(char_dir, "base_reference.png"); corrected_image = load_and_correct_image_orientation(image_file)
        if corrected_image: corrected_image.save(img_path, "PNG"); self.pm.add_character(name, img_path); st.toast(f"Character '{name}' created!", icon="👤"); st.rerun()
        else: st.error("Could not process image.")
        
    def create_voice(self, name: str, module_path: str, wav_file):
        voice_dir = os.path.join(self.pm.output_dir, "voices", name.replace(" ", "_")); os.makedirs(voice_dir, exist_ok=True)
        wav_path = os.path.join(voice_dir, "reference.wav");
        with open(wav_path, "wb") as f: f.write(wav_file.getbuffer())
        self.pm.add_voice(name, module_path, wav_path); st.toast(f"Voice '{name}' created!", icon="🗣️"); st.rerun()
    
    def delete_scene(self, scene_uuid: UUID):
        self.pm.delete_scene(scene_uuid); st.toast("Scene deleted.", icon="🗑️");
        select_item('project', self.pm.state.uuid); st.rerun()

    def reorder_scene(self, scene_uuid: UUID, direction: str):
        self.pm.reorder_scene(scene_uuid, direction); st.rerun()

    def reorder_shot(self, scene_uuid: UUID, shot_uuid: UUID, direction: str):
        self.pm.reorder_shot(scene_uuid, shot_uuid, direction); st.rerun()
    
    def add_shot_to_scene(self, scene_uuid: UUID, after_shot_uuid: UUID = None):
        new_shot_uuid = self.pm.add_shot_to_scene(scene_uuid, after_shot_uuid)
        st.toast("New shot added.", icon="🎞️")
        select_item('shot', (scene_uuid, new_shot_uuid))
        st.rerun()
    
    def delete_shot(self, scene_uuid: UUID, shot_uuid: UUID):
        self.pm.delete_shot(scene_uuid, shot_uuid); st.toast("Shot deleted.", icon="🗑️");
        select_item('scene', scene_uuid); st.rerun()
    
    def handle_shot_image_upload(self, scene_uuid: UUID, shot_uuid: UUID, image_file):
        shot_dir = os.path.join(self.pm.output_dir, "shots", str(shot_uuid)); os.makedirs(shot_dir, exist_ok=True)
        img_path = os.path.join(shot_dir, "uploaded_reference.png"); corrected_image = load_and_correct_image_orientation(image_file)
        if corrected_image:
            corrected_image.save(img_path, "PNG"); self.pm.update_shot(scene_uuid, shot_uuid, {"uploaded_image_path": img_path, "status": "upload_complete"}); st.toast("Uploaded image saved.");
        else: st.error("Could not process uploaded image.")

    def generate_shot_image(self, scene_uuid: UUID, shot_uuid: UUID): self._execute_and_reload("generate_shot_image", {"scene_uuid": scene_uuid, "shot_uuid": shot_uuid}, "Image generated!", "Image generation failed.")
    def generate_shot_video(self, scene_uuid: UUID, shot_uuid: UUID): self._execute_and_reload("generate_shot_video", {"scene_uuid": scene_uuid, "shot_uuid": shot_uuid}, "Video generated!", "Video generation failed.")
    def generate_shot_t2v(self, scene_uuid: UUID, shot_uuid: UUID): self._execute_and_reload("generate_shot_t2v", {"scene_uuid": scene_uuid, "shot_uuid": shot_uuid}, "T2V Video generated!", "T2V generation failed.")
    def generate_scene_audio(self, scene_uuid: UUID): self._execute_and_reload("generate_audio", {"scene_uuid": scene_uuid}, "Audio generated!", "Audio generation failed.")
    def assemble_final_video(self): self._execute_and_reload("assemble_final_video", {}, "Final video assembled!", "Final assembly failed.")
    def delete_character(self, uuid: UUID): self.pm.delete_character(uuid); st.toast("Character deleted.", icon="🗑️"); select_item('project', self.pm.state.uuid); st.rerun()
    def delete_voice(self, uuid: UUID): self.pm.delete_voice(uuid); st.toast("Voice deleted.", icon="🗑️"); select_item('project', self.pm.state.uuid); st.rerun()