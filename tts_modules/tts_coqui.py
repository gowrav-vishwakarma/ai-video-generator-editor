# tts_modules/tts_coqui.py
import os
import torch
import numpy as np
from typing import Tuple, Optional
from TTS.api import TTS as CoquiTTS
from moviepy import AudioFileClip
from scipy.io import wavfile

from base_modules import BaseTTS, BaseModuleConfig
from config_manager import DEVICE, clear_vram_globally

class CoquiTTSConfig(BaseModuleConfig):
    model_id: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    speaker_language: str = "en"

class CoquiTTSModule(BaseTTS):
    Config = CoquiTTSConfig

    def _load_model(self):
        if self.model is None:
            print(f"Loading TTS model: {self.config.model_id}...")
            self.model = CoquiTTS(model_name=self.config.model_id, progress_bar=True).to(DEVICE)
            print("TTS model loaded.")
    
    def clear_vram(self):
        print("Clearing TTS VRAM...")
        if self.model is not None:
            clear_vram_globally(self.model)
        self.model = None
        print("TTS VRAM cleared.")

    def generate_audio(
        self, text: str, output_dir: str, scene_idx: int, speaker_wav: Optional[str] = None
    ) -> Tuple[str, float]:
        self._load_model()
        
        print(f"Generating audio for scene {scene_idx}: \"{text[:50]}...\"")
        output_path = os.path.join(output_dir, f"scene_{scene_idx}_audio.wav")
        
        tts_kwargs = {"language": self.config.speaker_language, "file_path": output_path}
        
        if "xtts" in self.config.model_id.lower():
            if speaker_wav and os.path.exists(speaker_wav):
                tts_kwargs["speaker_wav"] = speaker_wav
            else:
                if speaker_wav: print(f"Warning: Speaker WAV {speaker_wav} not found. XTTS using default voice.")
        
        self.model.tts_to_file(text, **tts_kwargs)
        
        duration = 0.0
        try:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                with AudioFileClip(output_path) as audio_clip:
                    duration = audio_clip.duration + 0.1 # Small buffer
            else: raise ValueError("Audio file not generated or is empty.")
        except Exception as e:
            print(f"Error getting duration for {output_path}: {e}. Creating fallback.")
            samplerate = 22050 
            wavfile.write(output_path, samplerate, np.zeros(int(0.1 * samplerate), dtype=np.int16))
            duration = 0.1

        print(f"Actual audio duration for scene {scene_idx}: {duration:.2f}s")
        return output_path, duration