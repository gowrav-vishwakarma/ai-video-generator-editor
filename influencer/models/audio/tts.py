from TTS.api import TTS
import os
from typing import Optional, Dict, Any
from influencer.config import ContentConfig

def load_tts(model_name: str, device: str = "cuda"):
    """Load a TTS model"""
    print(f"Loading TTS model: {model_name}...")
    # Make sure the model is downloaded or it will download on first run
    return TTS(model_name=model_name, progress_bar=True).to(device)

def generate_audio(
    text: str, 
    tts_model: Any, 
    output_path: str, 
    speaker_wav: Optional[str] = None,
    language: str = "en"
) -> str:
    """Generate audio for given text using TTS model"""
    print(f"Generating audio for: {text}")
    print(f"TTS model name: {getattr(tts_model, 'model_name', None)}")
    print(f"Speaker wav: {speaker_wav}")
    
    # Check if model is XTTS
    is_xtts = "xtts" in getattr(tts_model, 'model_name', '').lower()
    
    if is_xtts and speaker_wav:
        tts_model.tts_to_file(text, file_path=output_path, speaker_wav=speaker_wav, language="en")
    elif is_xtts:
        tts_model.tts_to_file(text, file_path=output_path, speaker="random", language="en")
    else:
        tts_model.tts_to_file(text, file_path=output_path)
    
    print(f"Audio saved to {output_path}")
    return output_path

def generate_scene_audio(
    narration_scenes: list, 
    tts_model: Any, 
    config: ContentConfig
) -> list:
    """Generate audio for all scenes in narration"""
    audio_paths = []
    
    for i, scene in enumerate(narration_scenes):
        audio_file = os.path.join(config.output_dir, f"scene_{i}_audio.wav")
        generate_audio(
            scene["text"],
            tts_model,
            audio_file,
            speaker_wav=config.speaker_wav if "xtts" in config.tts_model.lower() else None
        )
        audio_paths.append(audio_file)
    
    return audio_paths 

speaker_reference_audio = "record_out.wav" # REQUIRED for XTTS voice cloning 