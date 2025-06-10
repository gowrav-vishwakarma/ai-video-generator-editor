# file: nicegui_app.py

import asyncio
from dataclasses import field
import os
import time
from typing import Any, Dict, List, Optional

# NiceGUI imports
from nicegui import app, background_tasks, binding, run, ui

# Backend imports
from config_manager import ContentConfig
from module_discovery import discover_modules
from project_manager import ProjectManager
from ui_task_executor import UITaskExecutor
from utils import load_and_correct_image_orientation

# --- 1. Global State and Initial Setup ---
discovered_modules = discover_modules()

@binding.bindable_dataclass
class AppState:
    current_view: str = 'project_selection'
    active_project: Optional[ProjectManager] = None
    active_executor: Optional[UITaskExecutor] = None
    is_processing: bool = False

@binding.bindable_dataclass
class DialogState:
    topic: str = ''
    video_format: str = 'Portrait'
    length: int = 20
    min_s: int = 2
    max_s: int = 5
    flow_choice: str = 'I2V'
    audio_content: Any = None
    language: str = 'en'
    # Character state
    char_name: str = ''
    char_image: Any = None
    characters: List[Dict] = field(default_factory=list)
    # Module state
    llm: Optional[str] = next((m['path'] for m in discovered_modules.get('llm', []) if m.get('path')), None)
    tts: Optional[str] = next((m['path'] for m in discovered_modules.get('tts', []) if m.get('path')), None)
    t2i: Optional[str] = next((m['path'] for m in discovered_modules.get('t2i', []) if m.get('path')), None)
    i2v: Optional[str] = next((m['path'] for m in discovered_modules.get('i2v', []) if m.get('path')), None)
    t2v: Optional[str] = next((m['path'] for m in discovered_modules.get('t2v', []) if m.get('path')), None)

app_state = AppState()

# --- 2. Helper Functions ---
def get_caps_from_path(mod_type: str, path: str) -> Optional[Dict[str, Any]]:
    if not path: return None
    for mod in discovered_modules.get(mod_type, []):
        if mod.get('path') == path:
            return mod['caps']
    return None

def load_project_and_switch_view(project_name: str):
    """Loads a project into the global state and switches the UI view."""
    project_manager = ProjectManager(f"modular_reels_output/{project_name}")
    if project_manager.load_project():
        app_state.active_project = project_manager
        app_state.active_executor = UITaskExecutor(project_manager)
        app_state.is_processing = False
        app_state.current_view = 'processing_dashboard'
    else:
        ui.notify(f"Failed to load project: {project_name}", type='negative')

# --- 3. Main Page and UI Definition ---

@ui.page('/')
def main_page():
    """Builds the main user interface."""

    # --- DIALOG DEFINITION ---
    with ui.dialog().props('persistent') as dialog, ui.card().style('min-width: 700px'):
        dialog_state = DialogState()
        ui.label('Create New Project').classes('text-h6')

        # Create clean, filtered option lists
        llm_options = [m['path'] for m in discovered_modules.get('llm', []) if m.get('path')]
        tts_options = [m['path'] for m in discovered_modules.get('tts', []) if m.get('path')]
        t2i_options = [m['path'] for m in discovered_modules.get('t2i', []) if m.get('path')]
        i2v_options = [m['path'] for m in discovered_modules.get('i2v', []) if m.get('path')]
        t2v_options = [m['path'] for m in discovered_modules.get('t2v', []) if m.get('path')]
        
        with ui.stepper().props('flat').classes('w-full') as stepper:
            with ui.step('Content'):
                ui.textarea('Video Topic').bind_value(dialog_state, 'topic').props('w-full')
                with ui.row():
                    ui.select(['Portrait', 'Landscape'], label='Format').bind_value(dialog_state, 'video_format')
                    ui.number('Length (s)', value=20).bind_value(dialog_state, 'length')
                with ui.row():
                    ui.number('Min Scenes', value=2).bind_value(dialog_state, 'min_s')
                    ui.number('Max Scenes', value=5).bind_value(dialog_state, 'max_s')
                with ui.stepper_navigation():
                    ui.button('Next', on_click=stepper.next)

            with ui.step('Models'):
                ui.label('Generation Flow')
                ui.radio({"I2V": "Image to Video", "T2V": "Text to Video"}).bind_value(dialog_state, 'flow_choice').props('inline')
                with ui.column().bind_visibility_from(dialog_state, 'flow_choice', value='I2V'):
                    ui.select(t2i_options, label="Image Model (T2I)").bind_value(dialog_state, 't2i').props('w-full')
                    ui.select(i2v_options, label="Image-to-Video Model (I2V)").bind_value(dialog_state, 'i2v').props('w-full')
                with ui.column().bind_visibility_from(dialog_state, 'flow_choice', value='T2V'):
                    ui.select(t2v_options, label="Text-to-Video Model (T2V)").bind_value(dialog_state, 't2v').props('w-full')
                
                ui.separator()
                ui.select(llm_options, label='Language Model (LLM)').bind_value(dialog_state, 'llm').props('w-full')
                tts_select = ui.select(tts_options, label='Text-to-Speech Model').bind_value(dialog_state, 'tts').props('w-full')
                @ui.refreshable
                def language_selection():
                    caps = get_caps_from_path('tts', dialog_state.tts)
                    if caps and caps.supported_tts_languages:
                        ui.select(caps.supported_tts_languages, label='Narration Language').bind_value(dialog_state, 'language')
                tts_select.on('update:model-value', language_selection.refresh)
                language_selection()
                with ui.stepper_navigation():
                    ui.button('Next', on_click=stepper.next)
                    ui.button('Back', on_click=stepper.previous).props('flat')

            with ui.step('Assets (Optional)'):
                ui.upload(label="Reference Speaker Audio (.wav)", on_upload=lambda e: setattr(dialog_state, 'audio_content', e.content.read())).props('w-full')
                
                ui.separator().classes('my-4')
                ui.label("Characters").classes('text-lg')
                
                @ui.refreshable
                def character_list():
                    for i, char in enumerate(dialog_state.characters):
                        with ui.row().classes('w-full items-center'):
                            ui.label(char['name']).classes('font-bold')
                            ui.button(icon='delete', on_click=lambda i=i: dialog_state.characters.pop(i) and character_list.refresh()).props('flat dense')
                
                character_list()
                
                with ui.row().classes('items-end'):
                    ui.input("Character Name").bind_value(dialog_state, 'char_name')
                    ui.upload(label="Character Image", auto_upload=True, on_upload=lambda e: setattr(dialog_state, 'char_image', e.content)).props('dense')
                
                def add_character():
                    if dialog_state.char_name and dialog_state.char_image:
                        dialog_state.characters.append({'name': dialog_state.char_name, 'image': dialog_state.char_image})
                        dialog_state.char_name = ''
                        dialog_state.char_image = None
                        character_list.refresh()
                    else:
                        ui.notify('Character name and image are required.', type='warning')
                ui.button("Add Character", on_click=add_character).props('outline')

                with ui.stepper_navigation():
                    ui.button('Back', on_click=stepper.previous).props('flat')

        # --- DIALOG ACTIONS ---
        with ui.row().classes('w-full justify-end'):
            ui.button('Cancel', on_click=dialog.close, color='negative')
            
            def handle_create_project():
                """Synchronous function to create the project structure."""
                if not dialog_state.topic:
                    ui.notify("Video Topic is required.", type='negative')
                    return
                # (Validation for models is the same as before)
                
                dialog.close()
                ui.notify(f"Creating project '{dialog_state.topic}'...", color='positive')

                name = "".join(c for c in dialog_state.topic.lower() if c.isalnum() or c in " ").replace(" ", "_")[:50]
                output_dir = f"modular_reels_output/{name}_{int(time.time())}"
                
                module_selections = {'llm': dialog_state.llm, 'tts': dialog_state.tts}
                use_svd = dialog_state.flow_choice == 'I2V'
                if use_svd:
                    module_selections.update({'t2i': dialog_state.t2i, 'i2v': dialog_state.i2v, 't2v': None})
                else:
                    module_selections.update({'t2i': None, 'i2v': None, 't2v': dialog_state.t2v})

                cfg = ContentConfig(
                    output_dir=output_dir, video_format=dialog_state.video_format,
                    target_video_length_hint=dialog_state.length, min_scenes=dialog_state.min_s,
                    max_scenes=dialog_state.max_s, use_svd_flow=use_svd,
                    module_selections=module_selections, language=dialog_state.language
                )
                
                pm = ProjectManager(output_dir)
                pm.initialize_project(dialog_state.topic, cfg)

                if dialog_state.audio_content:
                    relative_path = "speaker_audio.wav"
                    with open(os.path.join(output_dir, relative_path), "wb") as f: f.write(dialog_state.audio_content)
                    pm.set_speaker_audio(relative_path)
                
                for char in dialog_state.characters:
                    char_dir = os.path.join(output_dir, "characters", char['name'].replace(" ", "_"))
                    os.makedirs(char_dir, exist_ok=True)
                    ref_image_path = os.path.join(char_dir, "reference.png")
                    with open(ref_image_path, "wb") as f: f.write(char['image'].read())
                    pm.add_character({"name": char['name'], "reference_image_path": ref_image_path})

                # Navigate to the dashboard for the new project
                load_project_and_switch_view(os.path.basename(output_dir))

            ui.button('Create Project', on_click=handle_create_project, color='primary')


    # --- MAIN PAGE LAYOUT ---
    with ui.header(elevated=True).classes('items-center justify-between text-white bg-primary'):
        ui.label('AI Video Generation Pipeline').classes('text-h5')
        ui.button('Create New Project', on_click=dialog.open, icon='add').props('flat color=white')

    # --- Project Selection View ---
    with ui.column().bind_visibility_from(app_state, 'current_view', value='project_selection'):
        ui.label('Existing Projects').classes('text-h4 mt-4')
        project_base_dir = "modular_reels_output"
        projects = sorted([d for d in os.listdir(project_base_dir) if os.path.isdir(os.path.join(project_base_dir, d))]
                          ) if os.path.exists(project_base_dir) else []
        if not projects:
            ui.label("No projects found. Create one to get started!")
        else:
            with ui.list().props('bordered separator'):
                for project_dir in projects:
                    with ui.item():
                        with ui.item_section():
                            ui.item_label(project_dir.rsplit('_', 1)[0].replace('_', ' ').title())
                        with ui.item_section(side=True):
                            ui.button('Load', on_click=lambda p=project_dir: load_project_and_switch_view(p))

    # --- Processing Dashboard View ---
    with ui.column().bind_visibility_from(app_state, 'current_view', value='processing_dashboard'):
        @ui.refreshable
        def dashboard():
            if not app_state.active_project: return

            with ui.row().classes('w-full items-center justify-between'):
                with ui.row(align_items='center'):
                    ui.button(on_click=lambda: setattr(app_state, 'current_view', 'project_selection'), icon='arrow_back').props('flat')
                    ui.label(app_state.active_project.state.project_info.topic).classes('text-h4')
                ui.button("Generate Script", on_click=lambda: ui.notify("Script generation started..."), icon='play_arrow')

            # Remainder of dashboard UI will be built here
            ui.label("Dashboard is under construction.").classes('text-lg m-4')

        dashboard()

# Start the app with the important main guard
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="AI Video Generation Pipeline", storage_secret="a_secret_key_is_needed_for_storage")