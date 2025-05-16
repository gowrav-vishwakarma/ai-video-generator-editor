# main_script.py
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionXLPipeline, StableVideoDiffusionPipeline, AutoPipelineForText2Video
from diffusers.utils import load_image, export_to_video
from TTS.api import TTS # Coqui TTS
from moviepy.editor import * # VideoFileClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import gc # Garbage collection

# --- 0. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "instagram_content"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LLM paths (example, download these first)
LLM_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
# TTS model (XTTSv2 example)
TTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2" # or "tts_models/en/ljspeech/tacotron2-DDC"
# T2I model
T2I_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
T2I_REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
# I2V model (SVD)
I2V_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
# T2V model (alternative)
T2V_MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"

# --- 1. INITIALIZE MODELS (Load only when needed or keep loaded if VRAM allows) ---
def load_llm():
    print("Loading LLM...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16, # Use float16 for 4090
        device_map="auto", # Automatically uses GPU
        # load_in_8bit=True, # Optional: if VRAM is an issue with other models loaded
        # load_in_4bit=True, # Optional: for even smaller footprint
    )
    # Using pipeline for simplicity
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", max_new_tokens=512)

def load_tts():
    print("Loading TTS model...")
    # Make sure you have the model downloaded or it will download on first run
    # For XTTSv2, you might need to specify a speaker_wav for voice cloning
    return TTS(model_name=TTS_MODEL_ID, progress_bar=True).to(DEVICE)

def load_t2i_pipeline():
    print("Loading T2I pipeline (SDXL)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        T2I_MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(DEVICE)
    # Optional: Load refiner if you want to use it
    # refiner = DiffusionPipeline.from_pretrained(
    #     T2I_REFINER_ID, text_encoder_2=pipe.text_encoder_2, vae=pipe.vae,
    #     torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    # ).to(DEVICE)
    # return pipe, refiner
    return pipe, None # Simpler for now

def load_i2v_pipeline(): # SVD
    print("Loading I2V pipeline (SVD)...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        I2V_MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    ).to(DEVICE)
    pipe.enable_model_cpu_offload() # If VRAM is tight with other models
    return pipe

def load_t2v_pipeline(): # ModelScope Text-to-Video
    print("Loading T2V pipeline (ModelScope)...")
    pipe = AutoPipelineForText2Video.from_pretrained(
        T2V_MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    ).to(DEVICE)
    pipe.enable_model_cpu_offload()
    return pipe

# --- UTILITY TO CLEAR VRAM ---
def clear_vram(*models_or_pipelines):
    for item in models_or_pipelines:
        if hasattr(item, 'cpu') and callable(getattr(item, 'cpu')):
            item.cpu() # If it's a pipeline/model with a .cpu() method
        elif hasattr(item, 'model') and hasattr(item.model, 'cpu') and callable(getattr(item.model, 'cpu')):
            item.model.cpu()
    del models_or_pipelines
    torch.cuda.empty_cache()
    gc.collect()
    print("VRAM cleared and memory collected.")

# --- 2. TEXT GENERATION ---
def generate_script_and_prompts(topic, llm_pipeline):
    print(f"Generating script and prompts for topic: {topic}")
    # Improved prompt for structured output
    prompt = f"""
    You are an AI assistant creating content for an Instagram Reel about "{topic}".
    The Reel should be engaging and around 15-30 seconds long.
    Provide the following:
    1. Narration Script: A short, punchy narration. Break it into 2-3 short sentences or scenes.
    2. Visual Prompts: For each sentence/scene in the narration, provide a detailed visual prompt suitable for an image or video generation model. Describe the scene, style, colors, and mood.
    3. Hashtags: 3-5 relevant hashtags.

    Format your response clearly, for example:

    Narration Script:
    Scene 1: [Sentence 1 for narration]
    Scene 2: [Sentence 2 for narration]

    Visual Prompts:
    Scene 1: [Visual prompt for sentence 1, e.g., "A vibrant, futuristic cityscape at sunset, Blade Runner style, cinematic lighting, detailed."]
    Scene 2: [Visual prompt for sentence 2, e.g., "Close up on a curious robot eye, glowing blue, intricate details, macro shot."]

    Hashtags:
    #tag1 #tag2 #tag3
    """
    response = llm_pipeline(prompt, num_return_sequences=1)[0]['generated_text']
    print("LLM Response:\n", response)

    # Basic parsing (this needs to be robust)
    narration_parts = []
    visual_prompts_parts = []
    hashtags_str = ""

    in_narration = False
    in_visuals = False
    in_hashtags = False

    current_scene_narration = []
    current_scene_visuals = []

    for line in response.splitlines():
        line_lower = line.lower()
        if "narration script:" in line_lower:
            in_narration = True
            in_visuals = False
            in_hashtags = False
            continue
        elif "visual prompts:" in line_lower:
            in_narration = False
            in_visuals = True
            in_hashtags = False
            continue
        elif "hashtags:" in line_lower:
            in_narration = False
            in_visuals = False
            in_hashtags = True
            continue

        if in_narration and line.strip() and not line_lower.startswith("scene"):
            if line_lower.startswith("scene"): # Handles cases where scene numbering is on the same line
                 narration_parts.append(line.split(":",1)[1].strip())
            else:
                narration_parts.append(line.strip())
        elif in_visuals and line.strip() and not line_lower.startswith("scene"):
            if line_lower.startswith("scene"):
                visual_prompts_parts.append(line.split(":",1)[1].strip())
            else:
                visual_prompts_parts.append(line.strip())
        elif in_hashtags and line.strip():
            hashtags_str += line.strip() + " "

    # A bit of cleanup for parsing, assuming one line per scene now
    # This parsing is VERY basic and likely needs improvement based on LLM output format
    parsed_narration = [s.split(":", 1)[1].strip() for s in response.split("Narration Script:")[1].split("Visual Prompts:")[0].splitlines() if "Scene" in s]
    parsed_visuals = [s.split(":", 1)[1].strip() for s in response.split("Visual Prompts:")[1].split("Hashtags:")[0].splitlines() if "Scene" in s]
    parsed_hashtags = response.split("Hashtags:")[1].strip() if "Hashtags:" in response else "#GeneratedContent"


    if not parsed_narration or not parsed_visuals:
        print("Error: Could not parse LLM response properly. Using fallback.")
        # Fallback if parsing fails (common issue with LLM outputs)
        parsed_narration = [f"A short segment about {topic}.", "Concluding thoughts on the future."]
        parsed_visuals = [
            f"Cinematic shot of {topic}, vibrant colors, high detail, trending on artstation.",
            f"Abstract representation of {topic}, thought-provoking, professional."
        ]
        parsed_hashtags = f"#{topic.replace(' ', '')} #AI #GeneratedContent"

    return parsed_narration, parsed_visuals, parsed_hashtags.strip()


# --- 3. AUDIO GENERATION ---
def generate_audio(text, tts_model, output_path, speaker_wav=None): # speaker_wav for XTTS
    print(f"Generating audio for: {text}")
    if "xtts" in TTS_MODEL_ID.lower() and speaker_wav:
        tts_model.tts_to_file(text, speaker_wav=speaker_wav, language="en", file_path=output_path)
    else: # For other models like Tacotron
        tts_model.tts_to_file(text, file_path=output_path)
    print(f"Audio saved to {output_path}")
    return output_path

# --- 4. VISUAL GENERATION ---
# Option A: Image (SDXL) then Video (SVD) - Recommended for "Subject to Video"
def generate_image_then_video(image_prompt, i2v_pipe, t2i_pipe, refiner_pipe, scene_idx):
    # Generate Image (Keyframe)
    print(f"Generating keyframe image for: {image_prompt}")
    # For SDXL, you can add negative prompts, specify num_inference_steps, guidance_scale etc.
    # Using SDXL
    image = t2i_pipe(
        prompt=image_prompt,
        # negative_prompt="low quality, blurry, watermark", # Example
        num_inference_steps=30, # SDXL typically needs fewer steps
        guidance_scale=7.5,
        # If using refiner:
        # output_type="latent" if refiner_pipe else "pil",
    ).images[0]
    # if refiner_pipe:
    #     image = refiner_pipe(prompt=image_prompt, image=image[None, :]).images[0]

    image_path = os.path.join(OUTPUT_DIR, f"scene_{scene_idx}_keyframe.png")
    image.save(image_path)
    print(f"Keyframe image saved to {image_path}")

    # Generate Video from Image (SVD)
    print(f"Generating video from image using SVD...")
    # SVD parameters
    video_frames = i2v_pipe(
        image,
        decode_chunk_size=8, # Lower if OOM, max typically 8 or 4
        num_frames=25, # SVD XT default, ~4 seconds at low FPS
        motion_bucket_id=127, # Adjust for more/less motion
        fps=7, # Low FPS for SVD output, will be re-timed in MoviePy
        noise_aug_strength=0.02 # Default
    ).frames[0]

    video_clip_path = os.path.join(OUTPUT_DIR, f"scene_{scene_idx}_svd.mp4")
    export_to_video(video_frames, video_clip_path, fps=7) # Save with SVD's native FPS
    print(f"SVD video clip saved to {video_clip_path}")
    return video_clip_path

# Option B: Direct Text-to-Video (ModelScope)
def generate_direct_video(video_prompt, t2v_pipe, scene_idx):
    print(f"Generating direct video for: {video_prompt}")
    # ModelScope parameters
    video_frames = t2v_pipe(video_prompt, num_inference_steps=25, num_frames=20).frames # ~2-3 sec
    video_clip_path = os.path.join(OUTPUT_DIR, f"scene_{scene_idx}_t2v.mp4")
    export_to_video(video_frames, video_clip_path, fps=8) # ModelScope native FPS
    print(f"T2V video clip saved to {video_clip_path}")
    return video_clip_path

# --- 5. VIDEO ASSEMBLY ---
def assemble_final_video(video_clip_paths, audio_clip_paths, narration_parts, output_filename="final_reel.mp4"):
    print("Assembling final video...")
    final_clips = []
    target_resolution = (1080, 1920) # Instagram Reel 9:16

    for i, (video_path, audio_path, text_caption) in enumerate(zip(video_clip_paths, audio_clip_paths, narration_parts)):
        video_clip = VideoFileClip(video_path).resize(height=target_resolution[1]) # Resize to fit height
        if video_clip.w > target_resolution[0]: # If wider, crop
            video_clip = video_clip.crop(x_center=video_clip.w/2, width=target_resolution[0])
        video_clip = video_clip.set_position('center')


        audio_clip = AudioFileClip(audio_path)
        video_clip = video_clip.set_audio(audio_clip)
        video_clip = video_clip.set_duration(audio_clip.duration) # Sync video duration to audio

        # Add text overlay (simple example)
        txt_clip = TextClip(
            text_caption,
            fontsize=60,
            color='white',
            font='Arial-Bold', # Make sure font is available
            stroke_color='black',
            stroke_width=2,
            method='caption', # Wraps text
            size=(target_resolution[0]*0.8, None) # Width is 80% of video width
        )
        txt_clip = txt_clip.set_pos(('center', 0.8), relative=True).set_duration(video_clip.duration) # Position at 80% from top

        # Composite video with text
        scene_composite = CompositeVideoClip([video_clip, txt_clip], size=target_resolution)
        final_clips.append(scene_composite)

    if not final_clips:
        print("No clips to assemble!")
        return None

    final_video = concatenate_videoclips(final_clips, method="compose")
    final_video_path = os.path.join(OUTPUT_DIR, output_filename)
    final_video.write_videofile(final_video_path, fps=24, codec="libx264", audio_codec="aac") # Standard Instagram settings
    print(f"Final video saved to {final_video_path}")
    return final_video_path

# --- MAIN WORKFLOW ---
def main_automation_flow(topic, use_svd_flow=True):
    # --- Load models ---
    # Manage VRAM: Load one by one or use .cpu_offload()
    llm = None
    tts_model = None
    t2i_pipe, refiner = None, None
    i2v_pipe = None
    t2v_pipe = None
    final_video_path = None
    speaker_reference_audio = "path_to_your_reference_voice.wav" # REQUIRED for XTTS voice cloning
    # Make sure speaker_reference_audio exists or provide a default way to handle TTS without it.
    # e.g. if not os.path.exists(speaker_reference_audio): speaker_reference_audio = None

    try:
        # 1. LLM for Script and Prompts
        llm = load_llm()
        narration_scenes, visual_prompts_scenes, hashtags = generate_script_and_prompts(topic, llm)
        clear_vram(llm) # Unload LLM to free VRAM for vision models

        # 2. TTS for Narration
        tts_model = load_tts()
        audio_paths = []
        for i, scene_text in enumerate(narration_scenes):
            audio_file = os.path.join(OUTPUT_DIR, f"scene_{i}_audio.wav")
            # For XTTSv2, provide speaker_wav for voice cloning.
            # Ensure you have a reference audio file for XTTS if using it.
            # If speaker_reference_audio is not set, XTTS might use a default voice or fail.
            # For non-XTTS models, speaker_wav can be None.
            generate_audio(scene_text, tts_model, audio_file, speaker_wav=speaker_reference_audio if "xtts" in TTS_MODEL_ID.lower() else None)
            audio_paths.append(audio_file)
        clear_vram(tts_model)

        # 3. Visual Generation
        video_clip_paths = []
        if use_svd_flow: # Image (SDXL) -> Video (SVD)
            t2i_pipe, refiner = load_t2i_pipeline() # SDXL
            i2v_pipe = load_i2v_pipeline()         # SVD
            for i, visual_prompt in enumerate(visual_prompts_scenes):
                clip_path = generate_image_then_video(visual_prompt, i2v_pipe, t2i_pipe, refiner, i)
                video_clip_paths.append(clip_path)
            clear_vram(t2i_pipe, refiner, i2v_pipe)
        else: # Direct Text-to-Video (ModelScope)
            t2v_pipe = load_t2v_pipeline()
            for i, visual_prompt in enumerate(visual_prompts_scenes):
                clip_path = generate_direct_video(visual_prompt, t2v_pipe, i)
                video_clip_paths.append(clip_path)
            clear_vram(t2v_pipe)

        # 4. Video Assembly
        if video_clip_paths and audio_paths:
            final_video_path = assemble_final_video(video_clip_paths, audio_paths, narration_scenes)
        else:
            print("Not enough assets generated to assemble video.")

        # 5. Output final info
        if final_video_path:
            print("\n--- AUTOMATION COMPLETE ---")
            print(f"Final Video: {final_video_path}")
            print(f"Suggested Instagram Caption Text:\n{' '.join(narration_scenes)}")
            print(f"Suggested Hashtags: {hashtags}")
        else:
            print("\n--- AUTOMATION FAILED ---")
            print("Check logs for errors.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure all models are cleared from VRAM if they were loaded
        print("Cleaning up any remaining models from VRAM...")
        models_to_clear = [m for m in [llm, tts_model, t2i_pipe, refiner, i2v_pipe, t2v_pipe] if m is not None]
        if models_to_clear:
            clear_vram(*models_to_clear)
        print("Cleanup finished.")

if __name__ == "__main__":
    # --- IMPORTANT SETUP ---
    # 1. Install dependencies:
    #    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    #    pip install transformers accelerate bitsandbytes diffusers TTS moviepy Pillow safetensors sentencepiece
    #    (Make sure CUDA version in pytorch matches your system's CUDA Toolkit)
    # 2. Download models:
    #    The first time you run, Hugging Face models will download.
    #    For Coqui TTS, ensure you have the model files (XTTSv2 needs manual download or will try).
    # 3. For XTTSv2 voice cloning:
    #    Set `speaker_reference_audio` to a path of a clean WAV file of the voice you want to clone.
    #    It must be a 16-bit PCM WAV file, ideally >15 seconds long.
    # 4. Fonts for MoviePy: Make sure 'Arial-Bold' or your chosen font is available to MoviePy/ImageMagick.
    #    If not, ImageMagick might need to be installed and configured, or use a default font.

    # --- RUN THE SCRIPT ---
    # topic_for_reel = "The impact of AI on creative arts"
    topic_for_reel = "Exploring ancient underwater cities"
    # topic_for_reel = "The secrets of black holes"

    # Set to True for SDXL -> SVD flow (better "subject" control)
    # Set to False for direct Text-to-Video (ModelScope, simpler but maybe less coherent)
    use_image_to_video_flow = True

    # Ensure you have a speaker reference audio file if using XTTSv2
    # Create a dummy file for testing if you don't have one, but quality will be default.
    if not os.path.exists("path_to_your_reference_voice.wav") and "xtts" in TTS_MODEL_ID.lower():
        print("WARNING: XTTSv2 speaker reference audio not found. TTS might use a default voice or fail.")
        print("Please set 'speaker_reference_audio' in main_automation_flow to a valid .wav file.")
        # As a placeholder, you might allow it to run with default (if model supports) or skip audio:
        # speaker_reference_audio = None # This might make XTTSv2 use its default speaker.

    main_automation_flow(topic_for_reel, use_svd_flow=use_image_to_video_flow)