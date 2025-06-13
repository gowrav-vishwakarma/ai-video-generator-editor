# In ui_components.py
import streamlit as st
from system import get_discovered_modules, select_item
from utils import list_projects
import os

# ... other functions are unchanged ...
def render_project_selection_page(create_callback, load_callback):
    st.title("🎥 Modular AI Video Studio"); c1, c2 = st.columns([1, 2]);
    with c1:
        st.subheader("🚀 Create New Project")
        with st.form("new_project_form"):
            title = st.text_input("Project Title", "My Awesome Video"); video_format = st.selectbox("Format", ("Landscape", "Portrait"), index=0)
            if st.form_submit_button("Create Project", type="primary"):
                if title: create_callback(title, video_format)
    with c2:
        st.subheader("📂 Load Existing Project"); projects = list_projects();
        if not projects: st.info("No projects found.")
        else:
            for p in projects:
                with st.container(border=True):
                    cols = st.columns([4, 1]); cols[0].markdown(f"**{p['title']}**")
                    cols[1].button("Load", key=f"load_{p['name']}", on_click=load_callback, args=(p['name'],), use_container_width=True)

def render_asset_panel():
    st.subheader("🎭 Casting"); pm = st.session_state.current_project_manager; ui_executor = st.session_state.ui_executor
    with st.expander("Characters", expanded=True):
        if not pm.state.characters: st.caption("No characters created yet.")
        for char in pm.state.characters:
            cols = st.columns([1, 3, 1])
            if char.versions and char.versions[0].reference_image_path and os.path.exists(char.versions[0].reference_image_path):
                cols[0].image(char.versions[0].reference_image_path, width=50)
            cols[1].write(char.name)
            if cols[2].button("👁️", key=f"select_char_{char.uuid}", help="Inspect Character"): select_item('character', char.uuid)
        with st.popover("➕ Add New Character", use_container_width=True):
            with st.form("add_char_form", clear_on_submit=True):
                name = st.text_input("Character Name"); image = st.file_uploader("Upload Base Image", type=['png', 'jpg', 'jpeg'])
                if st.form_submit_button("Create Character", type="primary"):
                    if name and image: ui_executor.create_character(name, image); st.rerun()
    with st.expander("Voices", expanded=True):
        if not pm.state.voices: st.caption("No voices created yet.")
        for voice in pm.state.voices:
            cols = st.columns([4, 1]); cols[0].write(voice.name)
            if cols[1].button("👁️", key=f"select_voice_{voice.uuid}", help="Inspect Voice"): select_item('voice', voice.uuid)
        with st.popover("➕ Add New Voice", use_container_width=True):
            with st.form("add_voice_form", clear_on_submit=True):
                name = st.text_input("Voice Name"); tts_modules = get_discovered_modules().get('tts', []); tts_options = {mod['caps'].title: mod['path'] for mod in tts_modules}
                selected_title = st.selectbox("TTS Model", options=list(tts_options.keys())); speaker_wav = st.file_uploader("Reference Speaker Audio (.wav)", type=['wav'])
                if st.form_submit_button("Create Voice", type="primary"):
                    if name and selected_title and speaker_wav: ui_executor.create_voice(name, tts_options[selected_title], speaker_wav); st.rerun()

def render_viewer():
    st.subheader("📺 Viewer"); pm = st.session_state.current_project_manager
    selected_type = st.session_state.get('selected_item_type'); selected_uuid = st.session_state.get('selected_item_uuid')
    with st.container(border=True, height=450):
        item_to_show, video_path, image_path = None, None, None
        if selected_type == 'shot': item_to_show = pm.get_shot(*selected_uuid)
        elif selected_type == 'scene': item_to_show = pm.get_scene(selected_uuid)
        if item_to_show:
            video_path = getattr(item_to_show, 'video_path', None) or getattr(item_to_show, 'assembled_video_path', None)
            image_path = getattr(item_to_show, 'keyframe_image_path', None) or getattr(item_to_show, 'uploaded_image_path', None)
        if video_path and os.path.exists(video_path): st.video(video_path)
        elif image_path and os.path.exists(image_path): st.image(image_path, use_container_width=True)
        elif pm.state.final_video_path and os.path.exists(pm.state.final_video_path): st.video(pm.state.final_video_path)
        else: st.info("The final video or selected asset preview will appear here.")

def render_timeline():
    st.subheader("🗓️ Timeline"); pm = st.session_state.current_project_manager; ui_executor = st.session_state.ui_executor
    with st.container(border=True, height=300):
        if not pm.state.scenes: st.caption("No scenes yet. Use the Inspector to add a scene.")
        for i, scene in enumerate(pm.state.scenes):
            narration_duration = scene.narration.duration; total_shot_duration = sum(ui_executor.get_shot_duration(shot) for shot in scene.shots)
            if narration_duration == 0: summary_icon = "⏳"
            elif total_shot_duration >= narration_duration - 0.1: summary_icon = "✅"
            else: summary_icon = "❌"
            expander_title = f"{summary_icon} {scene.title} ({narration_duration:.1f}s)"
            with st.expander(expander_title, expanded=True):
                is_selected_scene = st.session_state.selected_item_type == 'scene' and st.session_state.selected_item_uuid == scene.uuid
                coverage_ratio = (total_shot_duration / narration_duration) if narration_duration > 0 else 1.0
                if narration_duration == 0: coverage_color, coverage_text = "grey", "Generate audio to see coverage"
                elif abs(total_shot_duration - narration_duration) < 0.1: coverage_color, coverage_text = "green", f"Coverage: {total_shot_duration:.1f}s / {narration_duration:.1f}s"
                elif coverage_ratio < 1.0: coverage_color, coverage_text = "red", f"Deficit: {(narration_duration - total_shot_duration):.1f}s"
                else: coverage_color, coverage_text = "orange", f"Surplus: {(total_shot_duration - narration_duration):.1f}s"
                st.progress(min(coverage_ratio, 1.0), text=coverage_text)
                st.markdown(f'<hr style="height:4px;border:none;color:{coverage_color};background-color:{coverage_color};" />', unsafe_allow_html=True)
                if st.button("Inspect Scene", key=f"select_scene_{scene.uuid}", type="primary" if is_selected_scene else "secondary", use_container_width=True): select_item('scene', scene.uuid)
                if scene.shots:
                    num_shots = len(scene.shots); shot_cols = st.columns(num_shots if num_shots > 0 else 1)
                    for j, shot in enumerate(scene.shots):
                        with shot_cols[j]:
                            is_selected_shot = st.session_state.selected_item_type == 'shot' and st.session_state.selected_item_uuid[1] == shot.uuid
                            status_icon = "✅" if shot.status == 'video_generated' else "🖼️" if shot.status in ['image_generated', 'upload_complete'] else "⏳"
                            if st.button(f"{status_icon} Shot {j+1}", key=f"select_shot_{shot.uuid}", use_container_width=True, type="primary" if is_selected_shot else "secondary"):
                                select_item('shot', (scene.uuid, shot.uuid))

def render_inspector_panel():
    selected_type = st.session_state.get('selected_item_type'); selected_uuid = st.session_state.get('selected_item_uuid')
    pm = st.session_state.current_project_manager; ui_executor = st.session_state.ui_executor
    if not selected_uuid: st.info("Select an item to inspect its properties."); return
    if selected_type == 'project':
        st.markdown(f"**Project: {pm.state.title}**"); st.write(f"Format: {pm.state.video_format}")
        st.checkbox("Add Narration Text to Final Video", value=pm.state.add_narration_text_to_video, key="project_add_text", on_change=lambda: setattr(pm.state, 'add_narration_text_to_video', st.session_state.project_add_text) or pm._save_state())
        st.divider(); st.subheader("Timeline Management"); st.button("➕ Add New Scene", use_container_width=True, on_click=ui_executor.add_scene)

    elif selected_type == 'scene':
        scene = pm.get_scene(selected_uuid); st.markdown(f"**🎬 Scene Inspector**")
        st.text_input("Scene Title", value=scene.title, key=f"title_{scene.uuid}", on_change=ui_executor.update_scene_title, args=(scene.uuid,))
        st.subheader("Narration"); st.text_area("Script", value=scene.narration.text, key=f"narration_{scene.uuid}", height=120, on_change=ui_executor.update_scene_narration, args=(scene.uuid,))
        
        voice_options = {v.name: v.uuid for v in pm.state.voices}
        if not voice_options: st.warning("No voices created!")
        else:
            # --- START: DEFINITIVE VOICE SELECTION FIX ---
            voice_names = list(voice_options.keys())
            
            # This logic now correctly handles setting the default on first render
            if not scene.narration.voice_uuid:
                scene.narration.voice_uuid = voice_options[voice_names[0]]
                pm._save_state()

            current_voice = pm.get_voice(scene.narration.voice_uuid)
            idx = voice_names.index(current_voice.name) if current_voice else 0
            
            # The on_change callback now correctly only takes the scene_uuid
            st.selectbox("Voice", options=voice_names, index=idx, key=f"voice_select_{scene.uuid}", on_change=ui_executor.update_scene_voice, args=(scene.uuid,))
            # --- END: DEFINITIVE VOICE SELECTION FIX ---

        can_gen_audio = bool(voice_options and scene.narration.text and scene.narration.voice_uuid)
        if scene.narration.audio_path and os.path.exists(scene.narration.audio_path):
            st.audio(scene.narration.audio_path)
            if st.button("Regen Audio", use_container_width=True, disabled=not can_gen_audio): ui_executor.generate_scene_audio(scene.uuid)
        else:
            if st.button("Gen Audio", use_container_width=True, type="primary", disabled=not can_gen_audio): ui_executor.generate_scene_audio(scene.uuid)
        st.divider(); st.subheader("Management")
        if st.button("➕ Add Shot", use_container_width=True): ui_executor.add_shot_to_scene(scene.uuid)
        if st.button("🗑️ Delete Scene", type="secondary", use_container_width=True, on_click=ui_executor.delete_scene, args=(scene.uuid,)): pass

    elif selected_type == 'shot':
        # This section is correct and unchanged
        scene_uuid, shot_uuid = selected_uuid; scene = pm.get_scene(scene_uuid); shot = pm.get_shot(scene_uuid, shot_uuid); shot_idx = scene.shots.index(shot)
        st.markdown(f"**🎞️ Shot {shot_idx + 1} Inspector**")
        with st.form(key=f"shot_form_{shot.uuid}"):
            st.subheader("Generation Flow"); flow_options = {"T2I ➡️ I2V": "T2I_I2V", "T2V": "T2V", "Upload ➡️ I2V": "Upload_I2V"}; flow_captions = list(flow_options.keys())
            current_flow_idx = flow_captions.index(next((k for k, v in flow_options.items() if v == shot.generation_flow), "T2I ➡️ I2V"))
            flow_selection = st.radio("Method", options=flow_captions, index=current_flow_idx, horizontal=True); selected_flow = flow_options[flow_selection]
            st.subheader("Direction & Casting"); visual = st.text_area("Visual Prompt", shot.visual_prompt, height=100)
            motion = st.text_area("Motion Prompt", shot.motion_prompt, height=68, help="For I2V models, describe desired motion.")
            char_options = {c.name: c.uuid for c in pm.state.characters}; default_chars = [pm.get_character(uid).name for uid in shot.character_uuids if pm.get_character(uid)]
            assigned_chars = st.multiselect("Characters", options=list(char_options.keys()), default=default_chars)
            st.subheader("Technical"); modules = get_discovered_modules(); new_module_selections = shot.module_selections.copy(); uploaded_file = None
            t2i_options = {m['caps'].title: m['path'] for m in modules.get('t2i', [])}; i2v_options = {m['caps'].title: m['path'] for m in modules.get('i2v', [])}; t2v_options = {m['caps'].title: m['path'] for m in modules.get('t2v', [])}
            t2i_paths_rev = {v: k for k, v in t2i_options.items()}; i2v_paths_rev = {v: k for k, v in i2v_options.items()}; t2v_paths_rev = {v: k for k, v in t2v_options.items()}
            new_t2i_title, new_i2v_title, new_t2v_title = None, None, None
            if selected_flow == "T2I_I2V":
                if t2i_options: current_t2i_path = shot.module_selections.get('t2i'); current_t2i_title = t2i_paths_rev.get(current_t2i_path, list(t2i_options.keys())[0]); new_t2i_title = st.selectbox("Image Model", options=t2i_options.keys(), index=list(t2i_options.keys()).index(current_t2i_title))
                if i2v_options: current_i2v_path = shot.module_selections.get('i2v'); current_i2v_title = i2v_paths_rev.get(current_i2v_path, list(i2v_options.keys())[0]); new_i2v_title = st.selectbox("Video Model", options=i2v_options.keys(), index=list(i2v_options.keys()).index(current_i2v_title))
            elif selected_flow == "T2V":
                if t2v_options: current_t2v_path = shot.module_selections.get('t2v'); current_t2v_title = t2v_paths_rev.get(current_t2v_path, list(t2v_options.keys())[0]); new_t2v_title = st.selectbox("T2V Model", options=t2v_options.keys(), index=list(t2v_options.keys()).index(current_t2v_title))
            elif selected_flow == "Upload_I2V":
                uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
                if i2v_options: current_i2v_path = shot.module_selections.get('i2v'); current_i2v_title = i2v_paths_rev.get(current_i2v_path, list(i2v_options.keys())[0]); new_i2v_title = st.selectbox("Video Model", options=i2v_options.keys(), index=list(i2v_options.keys()).index(current_i2v_title))
            if st.form_submit_button("💾 Update Shot Details"):
                if uploaded_file: ui_executor.handle_shot_image_upload(scene_uuid, shot_uuid, uploaded_file)
                final_module_selections = {}
                if new_t2i_title: final_module_selections['t2i'] = t2i_options.get(new_t2i_title)
                if new_i2v_title: final_module_selections['i2v'] = i2v_options.get(new_i2v_title)
                if new_t2v_title: final_module_selections['t2v'] = t2v_options.get(new_t2v_title)
                new_char_uuids = [char_options[name] for name in assigned_chars]
                ui_executor.pm.update_shot(scene_uuid, shot_uuid, {"generation_flow": selected_flow, "visual_prompt": visual, "motion_prompt": motion, "module_selections": final_module_selections, "character_uuids": new_char_uuids})
                st.toast("Shot details updated!"); st.rerun()
        st.subheader("Generation")
        if shot.generation_flow == "T2I_I2V":
            cols = st.columns(2)
            with cols[0]: st.button("🖼️ Generate Image", use_container_width=True, type="primary" if not shot.keyframe_image_path else "secondary", on_click=ui_executor.generate_shot_image, args=(scene_uuid, shot_uuid))
            with cols[1]: st.button("📹 Generate Video", use_container_width=True, type="primary" if not shot.video_path else "secondary", disabled=not shot.keyframe_image_path, on_click=ui_executor.generate_shot_video, args=(scene_uuid, shot_uuid))
        elif shot.generation_flow == "T2V": st.button("📹 Generate Video", use_container_width=True, type="primary", on_click=ui_executor.generate_shot_t2v, args=(scene_uuid, shot_uuid))
        elif shot.generation_flow == "Upload_I2V": st.button("📹 Generate Video from Upload", use_container_width=True, type="primary", disabled=not shot.uploaded_image_path, on_click=ui_executor.generate_shot_video, args=(scene_uuid, shot_uuid))
        st.divider(); st.button("🗑️ Delete Shot", type="secondary", use_container_width=True, on_click=ui_executor.delete_shot, args=(scene_uuid, shot.uuid))
    
    elif selected_type in ['character', 'voice']:
        asset = pm.get_character(selected_uuid) if selected_type == 'character' else pm.get_voice(selected_uuid)
        st.markdown(f"**{selected_type.title()} Inspector: {asset.name}**")
        if selected_type == 'voice':
            st.write(f"**TTS Model:** `{asset.tts_module_path.split('.')[-1]}`");
            if os.path.exists(asset.reference_wav_path): st.audio(asset.reference_wav_path)
        st.button(f"🗑️ Delete {selected_type.title()}", type="secondary", use_container_width=True, on_click=ui_executor.delete_character if selected_type == 'character' else ui_executor.delete_voice, args=(asset.uuid,))

    with st.expander("🐞 Debug: Show Full Project State"):
        st.json(pm.state.model_dump_json(indent=2))