# In app.py

import streamlit as st
import os
import json
from datetime import datetime
import torch
import time
from typing import List, Dict, Any

# Fix for Streamlit/Torch conflict
torch.classes.__path__ = []

# Local imports
from project_manager import ProjectManager
from config_manager import ContentConfig
from ui_task_executor import UITaskExecutor
from utils import load_and_correct_image_orientation
from module_discovery import discover_modules

# Page Config
st.set_page_config(page_title="AI Video Generation Pipeline", page_icon="🎥", layout="wide")

# Session State
def init_session_state():
    defaults = {
        'current_project': None, 
        'current_step': 'project_selection', 
        'auto_mode': True, 
        'ui_executor': None, 
        'speaker_audio': None, 
        'is_processing': False,
        'new_project_characters': [],
        'discovered_modules': discover_modules()
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
init_session_state()

# Helper Functions
def list_projects():
    projects = []
    base_dir = "modular_reels_output"
    if not os.path.exists(base_dir): return []
    for project_dir in os.listdir(base_dir):
        project_path = os.path.join(base_dir, project_dir)
        if os.path.isdir(project_path):
            project_file = os.path.join(project_path, "project.json")
            if os.path.exists(project_file):
                try:
                    with open(project_file, 'r') as f: data = json.load(f)
                    projects.append({'name': project_dir, 'topic': data['project_info']['topic'], 'created_at': datetime.fromtimestamp(data['project_info']['created_at']), 'status': data['project_info']['status']})
                except Exception as e:
                    st.error(f"Error loading project {project_dir}: {e}")
    return sorted(projects, key=lambda p: p['created_at'], reverse=True)


def go_to_step(step_name):
    st.session_state.current_step = step_name
    st.rerun()

def load_project(project_name):
    project_manager = ProjectManager(f"modular_reels_output/{project_name}")
    if project_manager.load_project():
        st.session_state.current_project = project_manager
        st.session_state.ui_executor = UITaskExecutor(project_manager)
        st.session_state.auto_mode = False; 
        st.session_state.is_processing = False
        speaker_relative_path = project_manager.state.project_info.speaker_audio_path
        if speaker_relative_path:
            # Construct the full, absolute path for the TTS module
            full_speaker_path = os.path.join(project_manager.output_dir, speaker_relative_path)
            if os.path.exists(full_speaker_path):
                st.session_state.speaker_audio = full_speaker_path
            else:
                st.session_state.speaker_audio = None
                st.warning(f"Saved speaker audio not found at: {full_speaker_path}")
        else:
            st.session_state.speaker_audio = None
        go_to_step('processing_dashboard')
    else:
        st.error("Failed to load project.")


def create_new_project(topic, auto, audio, video_format, length, min_s, max_s, use_svd, characters, module_selections, language):
    name = "".join(c for c in topic.lower() if c.isalnum() or c in " ").replace(" ", "_")[:50]
    output_dir = f"modular_reels_output/{name}_{int(time.time())}"
    
    cfg = ContentConfig(
        output_dir=output_dir, 
        aspect_ratio_format=video_format,
        target_video_length_hint=length, 
        min_scenes=min_s, 
        max_scenes=max_s, 
        use_svd_flow=use_svd,
        module_selections=module_selections,
        language=language
    )
    pm = ProjectManager(output_dir)
    pm.initialize_project(topic, cfg)

    if characters:
        for char_info in characters:
            safe_name = char_info['name'].replace(" ", "_")
            char_dir = os.path.join(output_dir, "characters", safe_name)
            os.makedirs(char_dir, exist_ok=True)
            ref_image_path = os.path.join(char_dir, "reference.png")
            
            corrected_image = load_and_correct_image_orientation(char_info['image'])
            if corrected_image:
                corrected_image.save(ref_image_path, "PNG") 
                pm.add_character({"name": char_info['name'], "reference_image_path": ref_image_path})
            else:
                st.error(f"Could not process image for character {char_info['name']}. Skipping.")
    
    st.session_state.current_project = pm
    st.session_state.ui_executor = UITaskExecutor(pm)
    st.session_state.auto_mode = auto
    if audio:
        # We will save a relative path to make the project folder more portable
        relative_speaker_path = "speaker_audio.wav"
        full_speaker_path = os.path.join(output_dir, relative_speaker_path)
        with open(full_speaker_path, "wb") as f: f.write(audio.getbuffer())
        
        # Save the full path to session state for the current run
        st.session_state.speaker_audio = full_speaker_path
        # Save the relative path to the project's permanent state
        pm.set_speaker_audio(relative_speaker_path)
        
    with st.spinner("Generating script..."):
        success = st.session_state.ui_executor.task_executor.execute_task("generate_script", {"topic": topic})

    if success:
        st.success("Script generated!")
        st.session_state.current_project.load_project()
        st.session_state.new_project_characters = []
        go_to_step('processing_dashboard')
    else:
        st.error("Failed to generate script.")
        st.session_state.current_project = None

# --- NEW: Callback function to handle flow changes ---
def handle_flow_change():
    """Clears character list when flow changes, as compatibility rules change."""
    st.session_state.new_project_characters = []

def render_project_selection():
    st.title("🎥 AI Video Generation Pipeline")
    
    def get_caps_from_path(mod_type: str, path: str) -> Dict[str, Any]:
        if not path: return None
        for mod in st.session_state.discovered_modules.get(mod_type, []):
            if mod['path'] == path:
                return mod['caps']
        return None

    c1, c2 = st.columns([1.2, 2])
    
    with c2:
        st.subheader("Existing Projects")
        for p in list_projects():
            with st.container(border=True):
                pc1, pc2, pc3 = st.columns([4, 2, 1])
                pc1.write(f"**{p['topic']}**"); pc2.write(f"_{p['created_at'].strftime('%Y-%m-%d %H:%M')}_")
                pc3.button("Load", key=f"load_{p['name']}", on_click=load_project, args=(p['name'],), use_container_width=True)
    
    with c1:
        st.subheader("Create New Project")
        
        st.info("Step 1: Choose your workflow and AI models.")
        st.radio(
            "Generation Flow",
            ("Image to Video (High Quality)", "Text to Video (Fast)"),
            horizontal=True,
            key="flow_choice",
            on_change=handle_flow_change
        )
        use_svd = st.session_state.flow_choice == "Image to Video (High Quality)"

        # --- FIX: MOVE THE DYNAMIC WIDGETS OUTSIDE THE FORM ---
        # The TTS selection needs to be outside the form so its on_change can work.
        tts_options = st.session_state.discovered_modules.get('tts', [])
        tts_paths = [m['path'] for m in tts_options]
        
        # We use st.session_state to store the selection.
        st.selectbox(
            "Text-to-Speech Model", 
            options=tts_paths, 
            format_func=lambda x: x.split('.')[-1],
            key="selected_tts_module", # The key ensures the selection persists across reruns
            on_change=lambda: st.session_state.update() 
        )

        # Get the capabilities of the selected TTS model from session_state
        selected_tts_caps = get_caps_from_path('tts', st.session_state.get('selected_tts_module'))
        
        # The language dropdown also depends on the TTS model, so it stays outside the form too.
        language = "en" # Default value
        if selected_tts_caps and selected_tts_caps.supported_tts_languages:
            supported_langs = selected_tts_caps.supported_tts_languages
            # We use a key here as well to persist the choice.
            language = st.selectbox("Narration Language", options=supported_langs, index=0, key="selected_language")
        elif selected_tts_caps:
            st.caption("Language selection not available for this model.")
        # --------------------------------------------------------

        with st.form("new_project_form"):
            has_characters = len(st.session_state.new_project_characters) > 0
            module_selections = {}

            # Now, inside the form, we just record the selections made outside.
            module_selections['tts'] = st.session_state.get('selected_tts_module')
            
            # Universal modules
            module_selections['llm'] = st.selectbox("Language Model (LLM)", options=[m['path'] for m in st.session_state.discovered_modules.get('llm', [])], format_func=lambda x: x.split('.')[-1])
            
            # (The original TTS selectbox is removed from here)

            # Workflow-specific selections
            show_char_section = False
            if use_svd:
                t2i_options = st.session_state.discovered_modules.get('t2i', [])
                if has_characters:
                    t2i_options = [m for m in t2i_options if m['caps'].supports_ip_adapter]

                if not t2i_options and has_characters:
                    st.error("No compatible Image Models (T2I) found for projects with characters.")
                    module_selections['t2i'] = None
                else:
                    module_selections['t2i'] = st.selectbox("Image Model (T2I)", options=[m['path'] for m in t2i_options], format_func=lambda x: x.split('.')[-1])
                
                module_selections['i2v'] = st.selectbox("Image-to-Video Model (I2V)", options=[m['path'] for m in st.session_state.discovered_modules.get('i2v', [])], format_func=lambda x: x.split('.')[-1])
                module_selections['t2v'] = st.session_state.discovered_modules.get('t2v', [{}])[0].get('path', None)
                
                t2i_caps = get_caps_from_path('t2i', module_selections.get('t2i'))
                if t2i_caps and t2i_caps.supports_ip_adapter:
                    show_char_section = True
                elif t2i_caps:
                    st.warning("This Image Model does not support characters.", icon="⚠️")
            
            else: # Text to Video Flow
                t2v_options = st.session_state.discovered_modules.get('t2v', [])
                if has_characters:
                    t2v_options = [m for m in t2v_options if m['caps'].supports_ip_adapter]

                if not t2v_options and has_characters:
                    st.error("No compatible Text-to-Video models found for projects with characters.")
                    module_selections['t2v'] = None
                else:
                    module_selections['t2v'] = st.selectbox("Text-to-Video Model (T2V)", options=[m['path'] for m in t2v_options], format_func=lambda x: x.split('.')[-1])

                module_selections['t2i'] = st.session_state.discovered_modules.get('t2i', [{}])[0].get('path', None)
                module_selections['i2v'] = st.session_state.discovered_modules.get('i2v', [{}])[0].get('path', None)
                
                t2v_caps = get_caps_from_path('t2v', module_selections.get('t2v'))
                if t2v_caps and t2v_caps.supports_ip_adapter:
                    show_char_section = True
                elif t2v_caps:
                    st.warning("This Text-to-Video model does not support characters.", icon="⚠️")

            st.divider()
            st.info("Step 2: Define your project topic and content.")
            topic = st.text_area("Video Topic")
            
            col1, col2 = st.columns(2)
            fmt = col1.selectbox("Format", ("Portrait", "Landscape"), index=0)
            length = col2.number_input("Length (s)", min_value=5, value=20, step=5)
            c1_s, c2_s = st.columns(2)
            min_s = c1_s.number_input("Min Scenes", 1, 10, 2, 1)
            max_s = c2_s.number_input("Max Scenes", min_s, 10, 5, 1)
            auto = st.checkbox("Automatic Mode", value=True)
            audio = st.file_uploader("Reference Speaker Audio (Optional, .wav)", type=['wav'])

            submitted = st.form_submit_button("Create & Start Project", type="primary")
            if submitted:
                # We now retrieve the language from the session state, where it was stored by the widget outside the form.
                final_language = st.session_state.get('selected_language', 'en') 

                if not all(module_selections.values()):
                    st.error("A required module is missing or could not be selected. Please check your selections.")
                elif not topic: 
                    st.error("Topic required.")
                else:
                    final_chars = st.session_state.new_project_characters if show_char_section else []
                    create_new_project(topic, auto, audio, fmt, length, min_s, max_s, use_svd, final_chars, module_selections, final_language) # Pass the language here
        
        st.divider()
        st.subheader("Add Characters (Optional)")
        if show_char_section:
            st.caption("Add characters to use for consistent generation.")
            for i, char in enumerate(st.session_state.new_project_characters):
                with st.container(border=True):
                    char_c1, char_c2 = st.columns([1, 4])
                    corrected_image = load_and_correct_image_orientation(char['image'])
                    if corrected_image: char_c1.image(corrected_image, width=64)
                    char_c2.write(f"**{char['name']}**")
            
            with st.expander("Add a New Character"):
                with st.form("add_character_form", clear_on_submit=True):
                    char_name = st.text_input("Character Name")
                    char_image = st.file_uploader("Upload Character Image", type=['png', 'jpg', 'jpeg'])
                    
                    if st.form_submit_button("Add Character to Project"):
                        if char_name and char_image:
                            st.session_state.new_project_characters.append({"name": char_name, "image": char_image})
                            st.rerun()
                        else:
                            st.warning("Character name and image are required.")
        else:
            st.info("The selected model workflow does not support character consistency. Please select a different model above to enable this feature.")
            if st.session_state.new_project_characters:
                st.session_state.new_project_characters = []


def render_processing_dashboard():
    project = st.session_state.current_project
    ui_executor = st.session_state.ui_executor

     # --- We need a more flexible add_scene_callback now ---
    def add_scene_at_callback(index_to_add):
        """Callback to add a new scene at a specific index."""
        st.session_state.ui_executor.add_new_scene(index_to_add)

    def remove_scene_callback(scene_idx_to_remove):
        """Callback to remove a specific scene."""
        st.session_state.ui_executor.remove_scene(scene_idx_to_remove)
    
    
    supports_characters = ui_executor.task_executor.active_flow_supports_characters
    use_svd_flow = project.state.project_info.config.get("use_svd_flow", True)

    st.title(f"🎬 Project: {project.state.project_info.topic}")
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        if st.button("⬅️ Back to Projects"): go_to_step('project_selection')
    with c2:
        if st.session_state.auto_mode:
            btn_text = "⏹️ Stop" if st.session_state.is_processing else "🚀 Start"
            if st.button(f"{btn_text} Automatic Processing", use_container_width=True, type="primary" if not st.session_state.is_processing else "secondary"):
                st.session_state.is_processing = not st.session_state.is_processing
    with c3:
        st.session_state.auto_mode = st.toggle("Automatic Mode", value=st.session_state.auto_mode, disabled=st.session_state.is_processing)
    st.divider()

    if supports_characters:
        with st.expander("👤 Project Characters & Subjects", expanded=False):
            if not project.state.characters:
                st.info("No characters defined. Add one to use features like IP-Adapter for consistency.")
            
            for char in project.state.characters:
                with st.container(border=True):
                    c1_char, c2_char = st.columns([1, 3])
                    with c1_char:
                        corrected_image = load_and_correct_image_orientation(char.reference_image_path)
                        if corrected_image: st.image(corrected_image, caption=char.name, use_container_width=True)
                    with c2_char:
                        with st.popover("Edit Character", use_container_width=True):
                            with st.form(f"edit_char_{char.name}"):
                                st.write(f"Editing: **{char.name}**")
                                new_name = st.text_input("New Name", value=char.name)
                                new_image = st.file_uploader("Upload New Image", type=['png', 'jpg', 'jpeg'], key=f"edit_img_{char.name}")
                                if st.form_submit_button("Save", type="primary"):
                                    ui_executor.update_character(char.name, new_name, new_image)
                        if st.button("Delete Character", key=f"del_char_{char.name}", type="secondary", use_container_width=True):
                            ui_executor.delete_character(char.name)

            with st.form("add_new_character_dashboard"):
                st.subheader("Add New Character")
                name = st.text_input("Character Name")
                image = st.file_uploader("Upload Reference Image", type=['png', 'jpg', 'jpeg'])
                if st.form_submit_button("Add Character", type="primary"):
                    if name and image:
                        ui_executor.add_character(name, image)
                    else: st.error("Name and image are required.")
    else:
        st.info("This project's workflow and selected models do not support character consistency (IP-Adapter).")

    st.subheader("Content Generation Dashboard")
    with st.expander("Reference Speaker Audio"):
        uploaded_file = st.file_uploader("Upload New Speaker Audio (.wav)", key="speaker_upload", disabled=st.session_state.is_processing)
        if uploaded_file:
            relative_speaker_path = "speaker_audio.wav"
            speaker_path = os.path.join(project.output_dir, relative_speaker_path)
            with open(speaker_path, "wb") as f: f.write(uploaded_file.getbuffer())
            
            st.session_state.speaker_audio = speaker_path
            # --- FIX: Update the project state when a new file is uploaded ---
            project.set_speaker_audio(relative_speaker_path)
            
            st.success("Speaker audio updated!")
            st.rerun() # Rerun to reflect the change
        if st.session_state.speaker_audio and os.path.exists(st.session_state.speaker_audio):
            st.write("Current audio:"); st.audio(st.session_state.speaker_audio)
        else:
            st.info("No reference audio provided.")

    next_task_name, next_task_data = project.get_next_pending_task()
    if (next_task_name == "assemble_final") or (next_task_name is None):
        if st.button("Assemble / View Final Video ➡️", type="primary"): go_to_step('video_assembly')
    st.write("---")

    # 1. "Add Scene at Beginning" button
    # We create a container to nicely center the button
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.button(
            "➕ Insert Scene Here",
            key="add_scene_at_0",
            on_click=add_scene_at_callback,
            args=(0,), # Add at index 0
            use_container_width=True,
            disabled=st.session_state.is_processing
        )

    for i, part in enumerate(project.state.script.narration_parts):
        with st.container(border=True):
            header_c1, header_c2 = st.columns([0.9, 0.1])
            with header_c1:
                st.header(f"Scene {i+1}")
            with header_c2:
                st.button(
                    "❌", 
                    key=f"delete_scene_{i}", 
                    help="Delete this scene", 
                    disabled=st.session_state.is_processing,
                    on_click=remove_scene_callback,
                    args=(i,) # Pass the scene index 'i' to the callback
                )
            if supports_characters:
                scene = project.get_scene_info(i)
                if scene and project.state.characters:
                    all_char_names = [c.name for c in project.state.characters]
                    selected_chars = st.multiselect(
                        "Characters in this Scene",
                        options=all_char_names,
                        default=scene.character_names,
                        key=f"scene_chars_{i}",
                        help="Select characters to feature. This will use their reference image for generation."
                    )
                    if selected_chars != scene.character_names:
                        ui_executor.update_scene_characters(i, selected_chars)
            
            st.subheader("Narration")
            new_text = st.text_area("Script", part.text, key=f"text_{i}", height=100, label_visibility="collapsed", disabled=st.session_state.is_processing)
            if new_text != part.text: ui_executor.update_narration_text(i, new_text)

            if part.audio_path and os.path.exists(part.audio_path):
                st.audio(part.audio_path)
                if st.button("Regen Audio", key=f"regen_audio_{i}", disabled=st.session_state.is_processing):
                    with st.spinner("..."): ui_executor.regenerate_audio(i, new_text, st.session_state.speaker_audio); st.rerun()
            else:
                if st.button("Gen Audio", key=f"gen_audio_{i}", disabled=st.session_state.is_processing):
                    with st.spinner("..."): ui_executor.regenerate_audio(i, new_text, st.session_state.speaker_audio); st.rerun()
            
            st.divider(); st.subheader("Visual Chunks")
            scene = project.get_scene_info(i)
            if scene:
                for chunk in scene.chunks:
                    chunk_idx = chunk.chunk_idx
                    with st.container(border=True):
                        if use_svd_flow:
                            p_col, i_col, v_col = st.columns([2, 1, 1])
                            with p_col:
                                st.write(f"**Chunk {chunk_idx + 1}**")
                                vis = st.text_area("Visual", chunk.visual_prompt, key=f"v_prompt_{i}_{chunk_idx}", height=125, disabled=st.session_state.is_processing)
                                if vis != chunk.visual_prompt: ui_executor.update_chunk_prompts(i, chunk_idx, visual_prompt=vis)
                                mot = st.text_area("Motion", chunk.motion_prompt, key=f"m_prompt_{i}_{chunk_idx}", height=75, disabled=st.session_state.is_processing)
                                if mot != chunk.motion_prompt: ui_executor.update_chunk_prompts(i, chunk_idx, motion_prompt=mot)
                            with i_col:
                                st.write("**Image**"); has_image = chunk.keyframe_image_path and os.path.exists(chunk.keyframe_image_path)
                                if has_image: st.image(chunk.keyframe_image_path)
                                else: st.info("Image pending...")
                                btn_txt = "Regen Image" if has_image else "Gen Image"
                                if st.button(btn_txt, key=f"gen_img_{i}_{chunk_idx}", disabled=st.session_state.is_processing, use_container_width=True):
                                    with st.spinner("..."): ui_executor.regenerate_chunk_image(i, chunk_idx); st.rerun()
                            with v_col:
                                st.write("**Video**"); has_video = chunk.video_path and os.path.exists(chunk.video_path)
                                if has_video: st.video(chunk.video_path)
                                else: st.info("Video pending...")
                                btn_txt = "Regen Video" if has_video else "Gen Video"
                                if st.button(btn_txt, key=f"gen_vid_{i}_{chunk_idx}", disabled=st.session_state.is_processing or not has_image, use_container_width=True):
                                    with st.spinner("..."): ui_executor.regenerate_chunk_video(i, chunk_idx); st.rerun()
                        else: # T2V Flow
                            p_col, v_col = st.columns([2, 1])
                            with p_col:
                                st.write(f"**Chunk {chunk_idx + 1} Prompt**")
                                vis = st.text_area("Prompt", chunk.visual_prompt, key=f"v_prompt_{i}_{chunk_idx}", height=125, disabled=st.session_state.is_processing)
                                if vis != chunk.visual_prompt: ui_executor.update_chunk_prompts(i, chunk_idx, visual_prompt=vis)
                            with v_col:
                                st.write("**Video**"); has_video = chunk.video_path and os.path.exists(chunk.video_path)
                                if has_video: st.video(chunk.video_path)
                                else: st.info("Video pending...")
                                btn_txt = "Regen Video" if has_video else "Gen Video"
                                if st.button(btn_txt, key=f"gen_t2v_{i}_{chunk_idx}", disabled=st.session_state.is_processing, use_container_width=True):
                                    with st.spinner("..."): ui_executor.regenerate_chunk_t2v(i, chunk_idx); st.rerun()
            elif part.status == "generated":
                 if st.button("Create Scene", key=f"create_scene_{i}", disabled=st.session_state.is_processing):
                    with st.spinner("..."): ui_executor.create_scene(i); st.rerun()
            else: st.info("Generate audio before scene creation.")
        
        # "Add Scene After This One" button
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.button(
                "➕ Insert Scene Here",
                key=f"add_scene_at_{i+1}",
                on_click=add_scene_at_callback,
                args=(i + 1,), # Add at the next index
                use_container_width=True,
                disabled=st.session_state.is_processing
            )
        
    st.divider()

    if st.session_state.auto_mode and st.session_state.is_processing:
        if next_task_name is None:
            st.session_state.is_processing = False; st.toast("✅ All tasks done!"); go_to_step('video_assembly')
        else:
            msg = f"Executing: {next_task_name.replace('_', ' ')} for Scene {next_task_data.get('scene_idx', 0) + 1}..."
            if "chunk" in next_task_name: msg = f"Executing: {next_task_name.replace('_', ' ')} for Scene {next_task_data.get('scene_idx', 0) + 1} / Chunk {next_task_data.get('chunk_idx', 0) + 1}..."
            with st.spinner(msg):
                if next_task_name == 'generate_audio': next_task_data['speaker_wav'] = st.session_state.speaker_audio
                success = st.session_state.ui_executor.task_executor.execute_task(next_task_name, next_task_data)
            if success:
                fresh_pm = ProjectManager(st.session_state.current_project.output_dir); fresh_pm.load_project()
                st.session_state.current_project = fresh_pm
                st.session_state.ui_executor = UITaskExecutor(fresh_pm)
                st.rerun()
            else:
                st.error(f"❌ Failed on: {next_task_name}. Stopping."); st.session_state.is_processing = False


def render_video_assembly():
    st.title("Final Video Assembly")
    project = st.session_state.current_project
    if st.button("⬅️ Back to Dashboard"): go_to_step('processing_dashboard')
    st.divider()
    final_path = project.state.final_video.path
    if final_path and os.path.exists(final_path):
        st.subheader("Final Video"); st.video(final_path)
        with st.expander("Details"):
            st.write("**Narration:**", project.state.final_video.full_narration_text)
            st.write("**Hashtags:**", ", ".join(project.state.final_video.hashtags))
            
    if st.button("Re-Assemble Final Video", type="primary"):
        with st.spinner("..."):
            if not all(s.status == 'completed' for s in project.state.scenes):
                for scene in project.state.scenes:
                    if scene.status != 'completed':
                        st.write(f"Assembling scene {scene.scene_idx+1}...")
                        st.session_state.ui_executor.task_executor.execute_task("assemble_scene", {"scene_idx": scene.scene_idx})
            
            st.write("Assembling final video...")
            success = st.session_state.ui_executor.assemble_final_video()
            
        if success: st.success("Assembled!")
        else: st.error("Failed.")


# Main App Router
if st.session_state.current_step == 'project_selection': render_project_selection()
elif st.session_state.current_project:
    if st.session_state.current_step == 'processing_dashboard': render_processing_dashboard()
    elif st.session_state.current_step == 'video_assembly': render_video_assembly()
else: go_to_step('project_selection')