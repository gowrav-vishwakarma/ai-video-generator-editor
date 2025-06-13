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
from utils import list_projects, load_and_correct_image_orientation
from module_discovery import discover_modules
# --- START OF MODIFICATION ---
# Import the new detection function
from system import SystemConfig, load_system_config, save_system_config, detect_system_specs
# --- END OF MODIFICATION ---

# Page Config
st.set_page_config(page_title="AI Video Generation Pipeline", page_icon="🎥", layout="wide")

# Session State
def init_session_state():
    system_config = load_system_config()
    
    defaults = {
        'current_project': None, 
        'current_step': 'system_config_setup' if not system_config else 'project_selection',
        'system_config': system_config,
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


def create_new_project(title, topic, auto, audio, video_format, length, min_s, max_s, use_svd, characters, module_selections, language, add_narration_text, seed):
    name = "".join(c for c in title.lower() if c.isalnum() or c in " ").replace(" ", "_")[:50]
    output_dir = f"modular_reels_output/{name}_{int(time.time())}"
    
    cfg = ContentConfig(
        output_dir=output_dir, 
        aspect_ratio_format=video_format,
        target_video_length_hint=length, 
        min_scenes=min_s, 
        max_scenes=max_s, 
        use_svd_flow=use_svd,
        module_selections=module_selections,
        language=language,
        add_narration_text_to_video=add_narration_text,
        seed=seed
    )
    pm = ProjectManager(output_dir)
    pm.initialize_project(title, topic, cfg)

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
        relative_speaker_path = "speaker_audio.wav"
        full_speaker_path = os.path.join(output_dir, relative_speaker_path)
        with open(full_speaker_path, "wb") as f: f.write(audio.getbuffer())
        st.session_state.speaker_audio = full_speaker_path
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

def handle_flow_change():
    st.session_state.new_project_characters = []

def render_system_config_setup():
    st.title("⚙️ System Configuration")
    st.info("First, let's specify your available system resources. This helps the pipeline select compatible AI models and prevent memory errors. This information will be saved locally in `system.json` for future use.")
    
    # --- START OF MODIFICATION ---
    # Call the detection function to get default values for the form
    detected_vram, detected_ram = detect_system_specs()
    # --- END OF MODIFICATION ---
    
    with st.form("system_config_form"):
        # --- START OF MODIFICATION ---
        # Use the detected values as the default for the number_input widgets
        vram = st.number_input("Available GPU VRAM (GB)", min_value=1.0, value=detected_vram, step=0.5, help="We've tried to detect this automatically. Please confirm or adjust.")
        ram = st.number_input("Available System RAM (GB)", min_value=1.0, value=float(detected_ram), step=1.0, help="We've tried to detect this automatically. Please confirm or adjust.")
        # --- END OF MODIFICATION ---
        
        submitted = st.form_submit_button("Save and Continue", type="primary")
        
        if submitted:
            save_system_config(vram, ram)
            st.session_state.system_config = SystemConfig(vram_gb=vram, ram_gb=ram)
            st.success("System configuration saved!")
            time.sleep(1) 
            go_to_step('project_selection')


def render_project_selection():
    st.title("🎥 AI Video Generation Pipeline")
    
    def filter_modules_by_resources(modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        system_config = st.session_state.system_config
        if not system_config:
            return [] 
        
        compatible_modules = []
        for mod in modules:
            caps = mod['caps']
            if caps.vram_gb_min <= system_config.vram_gb and caps.ram_gb_min <= system_config.ram_gb:
                compatible_modules.append(mod)
            else:
                print(f"Filtering out module '{caps.title}': Needs {caps.vram_gb_min}GB VRAM / {caps.ram_gb_min}GB RAM. Have {system_config.vram_gb}/{system_config.ram_gb}.")
        return compatible_modules

    def get_caps_from_path(mod_type: str, path: str) -> Dict[str, Any]:
        if not path: return None
        for mod in st.session_state.discovered_modules.get(mod_type, []):
            if mod['path'] == path:
                return mod['caps']
        return None
        
    def format_module_option(mod_type: str, path: str) -> str:
        caps = get_caps_from_path(mod_type, path)
        return caps.title if caps and caps.title else (path.split('.')[-1] if path else "Not Selected")

    c1, c2 = st.columns([1.2, 2])
    
    with c2:
        st.subheader("Existing Projects")
        
        projects = list_projects()
        if not projects:
            st.info("No projects found. Create one to get started!")

        for p in projects:
            with st.container(border=True):
                proj_c1, proj_c2 = st.columns([3, 1])
                with proj_c1:
                    st.markdown(f"**{p['title']}**")
                with proj_c2:
                    st.caption(f"_{p['created_at'].strftime('%Y-%m-%d %H:%M')}_")
                
                status_map = { "completed": "✅ Completed", "in_progress": "⚙️ In Progress", "failed": "❌ Failed" }
                display_status = status_map.get(p['status'], p['status'].title())
                
                info_parts = [ f"**Flow:** {p['flow']}", f"**Status:** {display_status}" ]
                if p['duration'] > 0: info_parts.append(f"**Duration:** {p['duration']:.1f}s")
                
                st.markdown(" | ".join(info_parts), help="Project details")
                
                with st.expander("Show Modules Used"):
                    modules_used = p.get('modules', {})
                    if not modules_used:
                        st.caption("Module info not available.")
                    else:
                        module_info_str = ""
                        llm_title = format_module_option('llm', modules_used.get('llm'))
                        tts_title = format_module_option('tts', modules_used.get('tts'))
                        
                        module_info_str += f"- **LLM:** {llm_title}\n"
                        module_info_str += f"- **TTS:** {tts_title}\n"

                        if p['flow'] == "Image-to-Video":
                            t2i_title = format_module_option('t2i', modules_used.get('t2i'))
                            i2v_title = format_module_option('i2v', modules_used.get('i2v'))
                            module_info_str += f"- **Image Model:** {t2i_title}\n"
                            module_info_str += f"- **Video Model:** {i2v_title}\n"
                        else: # Text-to-Video
                            t2v_title = format_module_option('t2v', modules_used.get('t2v'))
                            module_info_str += f"- **Video Model:** {t2v_title}\n"
                        
                        st.markdown(module_info_str)

                btn_c1, btn_c2 = st.columns(2)
                
                with btn_c1:
                    st.button("Load Project", key=f"load_{p['name']}", on_click=load_project, args=(p['name'],), use_container_width=True)
                
                with btn_c2:
                    if p['final_video_path']:
                        with st.popover("▶️ Play Video", use_container_width=True):
                            st.video(p['final_video_path'])
                    else:
                        st.button("▶️ Play Video", key=f"play_{p['name']}", disabled=True, use_container_width=True, help="Video not available or project not completed.")
    
    with c1:
        st.subheader("Create New Project")
        with st.container(border=True):
            st.markdown(f"**System Specs:** `{st.session_state.system_config.vram_gb}` GB VRAM | `{st.session_state.system_config.ram_gb}` GB RAM")
            if st.button("Change System Specs", key="change_specs"):
                go_to_step('system_config_setup')

        st.info("Step 1: Choose your workflow and AI models (filtered by your specs).")
        st.radio("Generation Flow", ("Image to Video (High Quality)", "Text to Video (Fast)"), horizontal=True, key="flow_choice", on_change=handle_flow_change)
        use_svd = st.session_state.flow_choice == "Image to Video (High Quality)"

        tts_options = filter_modules_by_resources(st.session_state.discovered_modules.get('tts', []))
        tts_paths = [m['path'] for m in tts_options]
        st.selectbox("Text-to-Speech Model", options=tts_paths, format_func=lambda path: format_module_option('tts', path), key="selected_tts_module", on_change=lambda: st.session_state.update())

        selected_tts_caps = get_caps_from_path('tts', st.session_state.get('selected_tts_module'))
        language = "en"
        if selected_tts_caps and selected_tts_caps.supported_tts_languages:
            supported_langs = selected_tts_caps.supported_tts_languages
            language = st.selectbox("Narration Language", options=supported_langs, index=0, key="selected_language")
        elif selected_tts_caps:
            st.caption("Language selection not available for this model.")

        with st.form("new_project_form"):
            has_characters = len(st.session_state.new_project_characters) > 0
            module_selections = {'tts': st.session_state.get('selected_tts_module')}
            
            llm_options_filtered = filter_modules_by_resources(st.session_state.discovered_modules.get('llm', []))
            module_selections['llm'] = st.selectbox("Language Model (LLM)", options=[m['path'] for m in llm_options_filtered], format_func=lambda path: format_module_option('llm', path))
            
            show_char_section = False
            
            selected_video_model_path = None
            if use_svd:
                all_t2i_options = filter_modules_by_resources(st.session_state.discovered_modules.get('t2i', []))
                t2i_options = [m for m in all_t2i_options if not has_characters or m['caps'].supports_ip_adapter]
                
                all_i2v_options_filtered = filter_modules_by_resources(st.session_state.discovered_modules.get('i2v', []))
                
                module_selections['t2i'] = st.selectbox("Image Model (T2I)", options=[m['path'] for m in t2i_options], format_func=lambda path: format_module_option('t2i', path), key="t2i_selection", help="Models are filtered based on your system specs and character support.")
                module_selections['i2v'] = st.selectbox("Image-to-Video Model (I2V)", options=[m['path'] for m in all_i2v_options_filtered], format_func=lambda path: format_module_option('i2v', path), help="Models are filtered based on your system specs.")

                selected_video_model_path = module_selections.get('t2i')
                if selected_video_model_path:
                    selected_caps = get_caps_from_path('t2i', selected_video_model_path)
                    if selected_caps and selected_caps.supports_ip_adapter:
                        show_char_section = True
            else: # T2V Flow
                all_t2v_options = filter_modules_by_resources(st.session_state.discovered_modules.get('t2v', []))
                t2v_options = [m for m in all_t2v_options if not has_characters or m['caps'].supports_ip_adapter]
                
                module_selections['t2v'] = st.selectbox("Text-to-Video Model (T2V)", options=[m['path'] for m in t2v_options], format_func=lambda path: format_module_option('t2v', path), key="t2v_selection", help="Models are filtered based on your system specs and character support.")
                
                selected_video_model_path = module_selections.get('t2v')
                if selected_video_model_path:
                    selected_caps = get_caps_from_path('t2v', selected_video_model_path)
                    if selected_caps and selected_caps.supports_ip_adapter:
                        show_char_section = True
            
            st.divider()
            st.info("Step 2: Define your project title and content topic.")
            title = st.text_input("Project Title", help="A user-friendly name for your project. This will be used for the folder name.")
            topic = st.text_area("Video Topic / Prompt", help="The main idea or prompt for the AI to generate the script.")
            col1, col2 = st.columns(2)
            fmt = col1.selectbox("Format", ("Portrait", "Landscape"), index=1)
            length = col2.number_input("Length (s)", min_value=5, value=20, step=5)
            c1_s, c2_s = st.columns(2)
            min_s = c1_s.number_input("Min Scenes", 1, 10, 2, 1)
            max_s = c2_s.number_input("Max Scenes", min_s, 10, 5, 1)
            
            st.divider()
            st.info("Step 3: Final Touches")
            seed = st.number_input("Image Generation Seed", min_value=-1, value=-1, step=1, help="-1 for a random seed, or any other number for a fixed seed.")
            auto = st.checkbox("Automatic Mode", value=True)
            audio = st.file_uploader(
                "Reference Speaker Audio (Required, .wav)", 
                type=['wav'],
                help="Upload a short .wav file of the desired voice. This is required to create a project."
            )
            add_narration_text = st.checkbox("Add Narration Text to Video", value=True, help="Renders the narration text as captions on the final video.")

            submitted = st.form_submit_button("Create & Start Project", type="primary")
            if submitted:
                final_language = st.session_state.get('selected_language', 'en')
                flow_is_valid = (use_svd and module_selections.get('t2i') and module_selections.get('i2v')) or \
                                (not use_svd and module_selections.get('t2v'))

                if not flow_is_valid or not module_selections.get('llm') or not module_selections.get('tts'):
                    st.error("A required module for the selected workflow is missing. Please check your selections.")
                elif not title:
                    st.error("Project Title is required.")
                elif not topic: 
                    st.error("Video Topic / Prompt is required.")
                elif not audio:
                    st.error("Reference Speaker Audio is required. Please upload a .wav file.")
                else:
                    final_chars = st.session_state.new_project_characters if show_char_section else []
                    create_new_project(title, topic, auto, audio, fmt, length, min_s, max_s, use_svd, final_chars, module_selections, final_language, add_narration_text, seed)
        
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
                        else: st.warning("Character name and image are required.")
        else:
            st.info("The selected model workflow does not support character consistency.")
            if st.session_state.new_project_characters: st.session_state.new_project_characters = []


def render_processing_dashboard():
    project = st.session_state.current_project
    ui_executor = st.session_state.ui_executor

    def add_scene_at_callback(index_to_add): st.session_state.ui_executor.add_new_scene(index_to_add)
    def remove_scene_callback(scene_idx_to_remove): st.session_state.ui_executor.remove_scene(scene_idx_to_remove)
    def regen_shots_callback(scene_idx_to_regen):
        with st.spinner(f"Regenerating shots for Scene {scene_idx_to_regen + 1}..."): 
            st.session_state.ui_executor.regenerate_scene_shots(scene_idx_to_regen)
        st.rerun()

    supports_characters = ui_executor.task_executor.active_flow_supports_characters
    use_svd_flow = project.state.project_info.config.get("use_svd_flow", True)

    st.title(f"🎬 Project: {project.state.project_info.title}")
    st.caption(f"LLM Topic: {project.state.project_info.topic}")
    
    with st.container(border=True):
        def get_module_title(mod_type: str, path: str) -> str:
            if not path: return "N/A"
            for mod in st.session_state.discovered_modules.get(mod_type, []):
                if mod['path'] == path:
                    return mod['caps'].title
            return path.split('.')[-1]

        config_dict = project.state.project_info.config
        modules = config_dict.get('module_selections', {})
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.caption("Project Settings")
            flow = "Image-to-Video" if config_dict.get('use_svd_flow', True) else "Text-to-Video"
            fmt = config_dict.get('aspect_ratio_format', 'N/A')
            length = config_dict.get('target_video_length_hint', 'N/A')
            st.markdown(f"**Flow:** {flow}<br>**Format:** {fmt}<br>**Length:** {length}s", unsafe_allow_html=True)
            
        with c2:
            st.caption("Core Models")
            llm_title = get_module_title('llm', modules.get('llm'))
            tts_title = get_module_title('tts', modules.get('tts'))
            st.markdown(f"**LLM:** {llm_title}<br>**TTS:** {tts_title}", unsafe_allow_html=True)

        with c3:
            st.caption("Video Generation Models")
            if config_dict.get('use_svd_flow', True):
                t2i_title = get_module_title('t2i', modules.get('t2i'))
                i2v_title = get_module_title('i2v', modules.get('i2v'))
                st.markdown(f"**Image:** {t2i_title}<br>**Video:** {i2v_title}", unsafe_allow_html=True)
            else:
                t2v_title = get_module_title('t2v', modules.get('t2v'))
                st.markdown(f"**Video:** {t2v_title}")

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
        expander_label = "👤 Project Characters & Subjects"
        if project.state.characters: expander_label = f"👤 Project Characters & Subjects: {', '.join([c.name for c in project.state.characters])}"
        with st.expander(expander_label, expanded=False):
            if not project.state.characters: st.info("No characters defined.")
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
                                if st.form_submit_button("Save", type="primary"): ui_executor.update_character(char.name, new_name, new_image)
                        if st.button("Delete Character", key=f"del_char_{char.name}", type="secondary", use_container_width=True): ui_executor.delete_character(char.name)
            with st.form("add_new_character_dashboard"):
                st.subheader("Add New Character")
                name = st.text_input("Character Name")
                image = st.file_uploader("Upload Reference Image", type=['png', 'jpg', 'jpeg'])
                if st.form_submit_button("Add Character", type="primary"):
                    if name and image: ui_executor.add_character(name, image)
                    else: st.error("Name and image are required.")
    else:
        st.info("This project's workflow does not support character consistency.")

    st.subheader("Content Generation Dashboard")

    with st.expander("Assembly & Export Settings"):
        cfg = ContentConfig(**project.state.project_info.config)
        
        c1, c2 = st.columns(2)
        current_text_setting = cfg.add_narration_text_to_video
        new_text_setting = c1.checkbox("Add Narration Text to Video", value=current_text_setting, help="Render the narration as captions. Requires re-assembly.")
        if new_text_setting != current_text_setting:
            ui_executor.update_project_config('add_narration_text_to_video', new_text_setting)

        current_seed = cfg.seed
        new_seed = c2.number_input("Image Seed", value=current_seed, min_value=-1, step=1, help="-1 for random. Changing this requires re-generating images.")
        if new_seed != current_seed:
            ui_executor.update_project_config('seed', new_seed)


    with st.expander("Reference Speaker Audio"):
        uploaded_file = st.file_uploader("Upload New Speaker Audio (.wav)", key="speaker_upload", disabled=st.session_state.is_processing)
        if uploaded_file:
            relative_speaker_path = "speaker_audio.wav"
            speaker_path = os.path.join(project.output_dir, relative_speaker_path)
            with open(speaker_path, "wb") as f: f.write(uploaded_file.getbuffer())
            st.session_state.speaker_audio = speaker_path
            project.set_speaker_audio(relative_speaker_path)
            st.success("Speaker audio updated!")
            st.rerun() 
        if st.session_state.speaker_audio and os.path.exists(st.session_state.speaker_audio):
            st.write("Current audio:"); st.audio(st.session_state.speaker_audio)
        else:
            st.info("No reference audio provided.")

    next_task_name, next_task_data = project.get_next_pending_task()
    is_ready_for_assembly = (next_task_name == "assemble_final")
    is_fully_complete = (next_task_name is None)

    if is_ready_for_assembly or is_fully_complete:
        if st.button("Assemble / View Final Video ➡️", type="primary"):
            if is_ready_for_assembly:
                with st.spinner("Assembling final video..."):
                    success = ui_executor.assemble_final_video()
                    if success: go_to_step('video_assembly')
            else:
                go_to_step('video_assembly')
    
    st.write("---")

    insert_c1, insert_c2, insert_c3 = st.columns([1, 1, 1])
    with insert_c2:
        st.button("➕ Insert Scene Here", key="add_scene_at_0", on_click=add_scene_at_callback, args=(0,), use_container_width=True, disabled=st.session_state.is_processing)

    for i, part in enumerate(project.state.script.narration_parts):
        with st.container(border=True):
            header_c1, header_c2 = st.columns([0.9, 0.1])
            with header_c1: st.header(f"Scene {i+1}")
            with header_c2: st.button("❌", key=f"delete_scene_{i}", help="Delete this scene", disabled=st.session_state.is_processing, on_click=remove_scene_callback, args=(i,))
            
            if supports_characters:
                scene = project.get_scene_info(i)
                if scene and project.state.characters:
                    all_char_names = [c.name for c in project.state.characters]
                    selected_chars = st.multiselect("Characters in this Scene", options=all_char_names, default=scene.character_names, key=f"scene_chars_{i}")
                    if selected_chars != scene.character_names: ui_executor.update_scene_characters(i, selected_chars)
            
            st.subheader("Narration")
            new_text = st.text_area("Script", part.text, key=f"text_{i}", height=100, label_visibility="collapsed", disabled=st.session_state.is_processing)
            if new_text != part.text: ui_executor.update_narration_text(i, new_text)

            audio_col1, audio_col2 = st.columns(2)
            if part.audio_path and os.path.exists(part.audio_path):
                audio_col1.audio(part.audio_path)
                if audio_col2.button("Regen Audio", key=f"regen_audio_{i}", disabled=st.session_state.is_processing, use_container_width=True):
                    with st.spinner("..."): ui_executor.regenerate_audio(i, new_text, st.session_state.speaker_audio); st.rerun()
            else:
                if audio_col1.button("Gen Audio", key=f"gen_audio_{i}", disabled=st.session_state.is_processing, use_container_width=True):
                    with st.spinner("..."): ui_executor.regenerate_audio(i, new_text, st.session_state.speaker_audio); st.rerun()
            
            st.divider()
            
            scene = project.get_scene_info(i)
            if scene:
                shots_header_c1, shots_header_c2 = st.columns([0.75, 0.25])
                with shots_header_c1: st.subheader("Visual Shots")
                with shots_header_c2: st.button("Regen Shots", key=f"regen_shots_{i}", on_click=regen_shots_callback, args=(i,), disabled=st.session_state.is_processing, use_container_width=True, help="Regenerate all visual and motion prompts for this scene.")
                
                for shot in scene.shots:
                    shot_idx = shot.shot_idx
                    with st.container(border=True):
                        if use_svd_flow:
                            p_col, i_col, v_col = st.columns([2, 1, 1])
                            with p_col:
                                st.write(f"**Shot {shot_idx + 1}**")
                                vis = st.text_area("Visual", shot.visual_prompt, key=f"v_prompt_{i}_{shot_idx}", height=125, disabled=st.session_state.is_processing)
                                if vis != shot.visual_prompt: ui_executor.update_shot_prompts(i, shot_idx, visual_prompt=vis)
                                mot = st.text_area("Motion", shot.motion_prompt, key=f"m_prompt_{i}_{shot_idx}", height=75, disabled=st.session_state.is_processing)
                                if mot != shot.motion_prompt: ui_executor.update_shot_prompts(i, shot_idx, motion_prompt=mot)
                            with i_col:
                                st.write("**Image**"); has_image = shot.keyframe_image_path and os.path.exists(shot.keyframe_image_path)
                                if has_image: st.image(shot.keyframe_image_path)
                                else: st.info("Image pending...")
                                if st.button("Regen Image" if has_image else "Gen Image", key=f"gen_img_{i}_{shot_idx}", disabled=st.session_state.is_processing, use_container_width=True):
                                    with st.spinner("..."): ui_executor.regenerate_shot_image(i, shot_idx); st.rerun()
                            with v_col:
                                st.write("**Video**"); has_video = shot.video_path and os.path.exists(shot.video_path)
                                if has_video: st.video(shot.video_path)
                                else: st.info("Video pending...")
                                if st.button("Regen Video" if has_video else "Gen Video", key=f"gen_vid_{i}_{shot_idx}", disabled=st.session_state.is_processing or not has_image, use_container_width=True):
                                    with st.spinner("..."): ui_executor.regenerate_shot_video(i, shot_idx); st.rerun()
                        else: # T2V Flow
                            p_col, v_col = st.columns([2, 1])
                            with p_col:
                                st.write(f"**Shot {shot_idx + 1} Prompt**")
                                vis = st.text_area("Prompt", shot.visual_prompt, key=f"v_prompt_{i}_{shot_idx}", height=125, disabled=st.session_state.is_processing)
                                if vis != shot.visual_prompt: ui_executor.update_shot_prompts(i, shot_idx, visual_prompt=vis)
                            with v_col:
                                st.write("**Video**"); has_video = shot.video_path and os.path.exists(shot.video_path)
                                if has_video: st.video(shot.video_path)
                                else: st.info("Video pending...")
                                if st.button("Regen Video" if has_video else "Gen Video", key=f"gen_t2v_{i}_{shot_idx}", disabled=st.session_state.is_processing, use_container_width=True):
                                    with st.spinner("..."): ui_executor.regenerate_shot_t2v(i, shot_idx); st.rerun()
            elif part.status == "generated":
                 if st.button("Define Visual Shots", key=f"create_scene_{i}", disabled=st.session_state.is_processing, use_container_width=True, help="Generates the visual and motion prompts for this scene based on its narration."):
                    with st.spinner("..."): ui_executor.create_scene(i); st.rerun()
            else: st.info("Generate audio before scene creation.")
        
        insert_c1, insert_c2, insert_c3 = st.columns([1, 1, 1])
        with insert_c2:
            st.button("➕ Insert Scene Here", key=f"add_scene_at_{i+1}", on_click=add_scene_at_callback, args=(i + 1,), use_container_width=True, disabled=st.session_state.is_processing)
        
    st.divider()

    if st.session_state.auto_mode and st.session_state.is_processing:
        if next_task_name is None:
            st.session_state.is_processing = False; st.toast("✅ All tasks done!"); go_to_step('video_assembly')
        else:
            msg = f"Executing: {next_task_name.replace('_', ' ')} for Scene {next_task_data.get('scene_idx', 0) + 1}..."
            if "shot" in next_task_name: msg += f" / Shot {next_task_data.get('shot_idx', 0) + 1}"
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
            
        if success:
            st.success("Assembled!")
            st.rerun() 
        else:
            st.error("Failed.")

# Main application router
if st.session_state.current_step == 'system_config_setup':
    render_system_config_setup()
elif st.session_state.current_step == 'project_selection':
    render_project_selection()
elif st.session_state.current_project:
    if st.session_state.current_step == 'processing_dashboard':
        render_processing_dashboard()
    elif st.session_state.current_step == 'video_assembly':
        render_video_assembly()
else:
    go_to_step('project_selection')