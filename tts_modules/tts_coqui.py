# tts_modules/tts_coqui.py
import os
import torch
import numpy as np
from typing import Tuple, Optional
from TTS.api import TTS as CoquiTTS
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from moviepy.audio.AudioClip import concatenate_audioclips, AudioClip 
from moviepy.video.fx.Crop import Crop # Correct import based on your docs

from scipy.io import wavfile # For fallback silent audio
from config_manager import TTSConfig, DEVICE, clear_vram_globally

TTS_MODEL = None

def load_model(config: TTSConfig):
    global TTS_MODEL
    if TTS_MODEL is None:
        print(f"Loading TTS model: {config.model_id}...")
        TTS_MODEL = CoquiTTS(model_name=config.model_id, progress_bar=True).to(DEVICE)
        print("TTS model loaded.")
    return TTS_MODEL

def clear_tts_vram():
    global TTS_MODEL
    print("Clearing TTS VRAM...")
    models_to_clear = []
    if TTS_MODEL is not None:
        # Coqui TTS objects don't always have a .cpu() method directly on the main object
        # but components might. For now, direct deletion and gc is the primary method.
        # If specific offloading methods are known for Coqui TTS, they can be added.
        models_to_clear.append(TTS_MODEL) 
    
    clear_vram_globally(*models_to_clear)
    TTS_MODEL = None
    print("TTS VRAM cleared.")

def generate_audio(
    text: str, 
    output_dir: str, 
    scene_idx: int, 
    tts_config: TTSConfig,
    speaker_wav: Optional[str] = None
) -> Tuple[str, float]:
    model = load_model(tts_config) # Ensures model is loaded
    
    print(f"Generating audio for scene {scene_idx}: \"{text[:50]}...\"")
    output_path = os.path.join(output_dir, f"scene_{scene_idx}_audio.wav")
    
    tts_kwargs = {"language": tts_config.speaker_language, "file_path": output_path}
    
    if "xtts" in tts_config.model_id.lower():
        if speaker_wav and os.path.exists(speaker_wav):
            tts_kwargs["speaker_wav"] = speaker_wav
        else:
            if speaker_wav: print(f"Warning: Speaker WAV {speaker_wav} not found. XTTS using default voice.")
            else: print("Warning: XTTS model, but no speaker_wav. Using default voice.")
    
    model.tts_to_file(text, **tts_kwargs)
    print(f"Audio for scene {scene_idx} saved to {output_path}")
    
    duration = 0.0
    try:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with AudioFileClip(output_path) as audio_clip:
                duration = audio_clip.duration
            duration += 0.1 # Small buffer often good for TTS
        else:
            raise ValueError("Audio file not generated or is empty.")
    except Exception as e:
        print(f"Error getting duration for {output_path}: {e}. Creating fallback silent audio.")
        samplerate = 22050 
        silence_data = np.zeros(int(0.1 * samplerate), dtype=np.int16)
        wavfile.write(output_path, samplerate, silence_data)
        duration = 0.1
        print(f"Wrote fallback silent audio to {output_path}")

    print(f"Actual audio duration for scene {scene_idx}: {duration:.2f}s")
    return output_path, duration