# llm_modules/llm_zephyr.py
import torch
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from config_manager import ContentConfig, DEVICE, clear_vram_globally # Use global clear for now

@dataclass
class LLMConfig:
    """Configuration for Zephyr LLM model."""
    model_id: str = "HuggingFaceH4/zephyr-7b-beta"
    max_new_tokens_script: int = 1536
    max_new_tokens_chunk_prompt: int = 256 # Shorter for chunk prompts
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

MODEL = None
TOKENIZER = None

def load_model_and_tokenizer(config: LLMConfig):
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        print(f"Loading LLM: {config.model_id}...")
        TOKENIZER = AutoTokenizer.from_pretrained(config.model_id)
        if TOKENIZER.pad_token is None:
            TOKENIZER.pad_token = TOKENIZER.eos_token
        
        # Try to load with device_map="auto" first for multi-GPU or large models
        try:
            MODEL = AutoModelForCausalLM.from_pretrained(
                config.model_id, torch_dtype=torch.float16, device_map="auto"
            )
        except Exception as e:
            print(f"Failed to load LLM with device_map='auto' ({e}), trying with explicit device: {DEVICE}")
            MODEL = AutoModelForCausalLM.from_pretrained(
                config.model_id, torch_dtype=torch.float16
            ).to(DEVICE)
        print("LLM loaded.")
    return MODEL, TOKENIZER

def clear_llm_vram():
    global MODEL, TOKENIZER
    print("Clearing LLM VRAM...")
    # Models loaded with device_map="auto" don't need explicit .cpu() before del
    # but it doesn't hurt to try.
    models_to_clear = []
    if MODEL is not None:
        models_to_clear.append(MODEL)
    # Tokenizer is usually small, but good practice
    # if TOKENIZER is not None: models_to_clear.append(TOKENIZER) # Tokenizer isn't a model
    
    clear_vram_globally(*models_to_clear)
    MODEL = None
    TOKENIZER = None # Technically tokenizer doesn't consume much VRAM but good to reset
    print("LLM VRAM cleared.")


def _parse_llm_json_response(decoded_output: str, context: str = "script") -> Optional[Dict]:
    match = re.search(r'\{[\s\S]*\}', decoded_output)
    if match:
        json_text_to_parse = match.group(0)
    else:
        json_text_to_parse = decoded_output
        print(f"Warning: Could not extract a clear JSON block for {context}, attempting to parse full output.")

    try:
        json_text_to_parse = re.sub(r',(\s*[}\]])', r'\1', json_text_to_parse) # Trailing commas
        response_data = json.loads(json_text_to_parse)
        return response_data
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM JSON for {context}: {e}. Raw output was:\n{decoded_output}")
        return None

def generate_script(
    topic: str, 
    content_config: ContentConfig, 
    llm_config: LLMConfig
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    model, tokenizer = load_model_and_tokenizer(llm_config)
    print(f"Generating script and prompts for topic: {topic}")
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant creating content for a short video. "
                "Your response must be in valid JSON format: "
                "{\"narration\": [{\"scene\": 1, \"text\": \"narration_text\", \"duration_estimate\": float_seconds}], "
                "\"visuals\": [{\"scene\": 1, \"prompt\": \"visual_prompt_for_scene\"}], "
                "\"hashtags\": [\"tag1\", \"tag2\"]}"
            )
        },
        {
            "role": "user",
            "content": f"""
            Create content for a short video about "{topic}".
            The video should be engaging and the total narration should be around {content_config.target_video_length_hint} seconds long.
            Each narration segment should ideally be around {content_config.max_scene_narration_duration_hint} seconds.
            Generate between {content_config.min_scenes} and {content_config.max_scenes} narration/visual scenes.
            
            Return your response in this exact JSON format:
            {{
                "narration": [
                    {{"scene": 1, "text": "First scene narration text.", "duration_estimate": {content_config.max_scene_narration_duration_hint}}},
                    {{"scene": 2, "text": "Second scene narration text.", "duration_estimate": {content_config.max_scene_narration_duration_hint}}}
                ],
                "visuals": [
                    {{"scene": 1, "prompt": "Detailed visual prompt for scene 1, focusing on key elements, style (e.g., cinematic, anime, realistic), camera angles, and mood."}},
                    {{"scene": 2, "prompt": "Detailed visual prompt for scene 2, similarly descriptive."}}
                ],
                "hashtags": ["relevantTag1", "relevantTag2", "relevantTag3"]
            }}
            
            Ensure the 'duration_estimate' is a float representing seconds.
            Each visual prompt should be detailed and suitable for image/video generation models.
            The number of narration entries must match the number of visual entries.
            """
        }
    ]

    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "input_ids": tokenized_chat, "max_new_tokens": llm_config.max_new_tokens_script, 
        "do_sample": True, "top_k": llm_config.top_k, "top_p": llm_config.top_p, 
        "temperature": llm_config.temperature, "pad_token_id": tokenizer.eos_token_id
    }

    outputs = model.generate(**generation_kwargs)
    decoded_output = tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)
    print("LLM Script Response:\n", decoded_output)
    
    response_data = _parse_llm_json_response(decoded_output, "script")

    if response_data:
        try:
            # Validate structure (simplified from original for brevity, can be expanded)
            if not all(k in response_data for k in ["narration", "visuals", "hashtags"]):
                 raise ValueError("Missing required keys in response")
            
            narration_data = sorted(response_data.get("narration", []), key=lambda x: x["scene"])
            visuals_data = sorted(response_data.get("visuals", []), key=lambda x: x["scene"])
            
            narration_scenes = [{"text": s["text"], "duration_estimate": float(s.get("duration_estimate", content_config.max_scene_narration_duration_hint))} for s in narration_data]
            visual_prompts = [s["prompt"] for s in visuals_data]
            hashtags = response_data.get("hashtags", [])

            if len(narration_scenes) != len(visual_prompts):
                raise ValueError("Mismatch between number of narration and visual scenes.")
            if not (content_config.min_scenes <= len(narration_scenes) <= content_config.max_scenes):
                 print(f"Warning: Scene count {len(narration_scenes)} outside range [{content_config.min_scenes}, {content_config.max_scenes}]. Adjusting...")
                 # Could add logic here to trim/pad, or let it proceed. For now, just warn.

            return narration_scenes, visual_prompts, hashtags
        except (ValueError, KeyError) as e:
            print(f"Error processing parsed LLM script data: {e}. Using fallback.")
    
    # Fallback
    fallback_duration = content_config.max_scene_narration_duration_hint
    narration_scenes = [
        {"text": f"An introduction to {topic}.", "duration_estimate": fallback_duration},
        {"text": "Exploring details and implications.", "duration_estimate": fallback_duration}
    ]
    visual_prompts = [
        f"Cinematic overview of {topic}, vibrant colors, high detail.",
        f"Close-up abstract representation related to {topic}, thought-provoking."
    ]
    hashtags = [f"#{topic.replace(' ', '')}", "#AIvideo", "#GeneratedContent"]
    return narration_scenes, visual_prompts, hashtags


def generate_chunk_visual_prompts(
    scene_narration: str,
    original_scene_prompt: str,
    num_chunks: int,
    content_config: ContentConfig,
    llm_config: LLMConfig
) -> List[Tuple[str, str]]:  # Now returns tuple of (visual_prompt, motion_prompt)
    model, tokenizer = load_model_and_tokenizer(llm_config) # Ensures model is loaded
    print(f"Generating {num_chunks} chunk-specific visual prompts for scene...")
    
    chunk_prompts = []
    
    for chunk_idx in range(num_chunks):
        chunk_start_time = chunk_idx * content_config.model_max_video_chunk_duration
        # A simple way to estimate which part of narration this chunk covers
        narration_words = scene_narration.split()
        total_narration_words = len(narration_words)
        
        # Estimate words per chunk based on num_chunks and total words
        words_per_chunk_ideal = total_narration_words / num_chunks
        start_word_idx = int(chunk_idx * words_per_chunk_ideal)
        end_word_idx = int((chunk_idx + 1) * words_per_chunk_ideal)
        current_narration_segment = " ".join(narration_words[start_word_idx:end_word_idx])

        previous_prompt_context = f'Previous chunk showed: "{chunk_prompts[-1][0]}"' if chunk_prompts else 'This is the first chunk of the scene.'

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant creating concise visual prompts for short video chunks (around 2-3 seconds). "
                    "Make sure to keep it stick to the original topic, scene prompt, elements, characters and envioroment."
                    "Always pass the style and the mood of the original scene prompt to your response as it will be seen in isolation and still we want to keep the style and the mood of the original scene prompt."
                    "Visual prompt should describe the content, subject and environment in very details that we want ot show in the chunk. not just a general description of the scene"
                    "if required tell more details about subject like race age gender, if element then tell more details about type, look, material, color, etc."
                    "Do not use Text Overlays or text content in any prompt"
                    "Each prompt should describe a static image and no motion or video related description in visual prompt."
                    "For each chunk, you need to generate TWO prompts:\n"
                    "1. A visual prompt describing what to show (for image generation)\n"
                    "2. A motion prompt describing how the scene should move/transition (for video generation)\n"
                    "Keep both prompts under 77 tokens. Ensure visual continuity with the previous chunk and original style."
                    "Respond in JSON: {\"visual_prompt\": \"your_visual_prompt\", \"motion_prompt\": \"your_motion_prompt\"}"
                )
            },
            {
                "role": "user",
                "content": f"""
                Original scene visual prompt: "{original_scene_prompt}"
                Narration for this scene: "{scene_narration}"
                This specific chunk ({chunk_idx + 1}/{num_chunks}) should visually represent: "{current_narration_segment if current_narration_segment else 'a continuation or a general aspect of the scene'}"
                {previous_prompt_context}
                
                Generate TWO concise prompts for this chunk:
                1. A visual prompt describing what to show (for image generation)
                2. A motion prompt describing how the scene should move/transition (for video generation)
                
                Return your response in this exact JSON format:
                {{
                    "visual_prompt": "A short, focused visual prompt (max 77 tokens) for this 2-3 second chunk, consistent with original style and previous chunk.",
                    "motion_prompt": "A short description of how the scene should move or transition (max 77 tokens), e.g., 'slow pan left', 'zoom in', 'smooth transition', etc."
                }}
                """
            }
        ]
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        generation_kwargs = {
            "input_ids": tokenized_chat, "max_new_tokens": llm_config.max_new_tokens_chunk_prompt, 
            "do_sample": True, "top_k": llm_config.top_k, "top_p": llm_config.top_p, 
            "temperature": llm_config.temperature, "pad_token_id": tokenizer.eos_token_id
        }
        outputs = model.generate(**generation_kwargs)
        decoded_output = tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)
        
        response_data = _parse_llm_json_response(decoded_output, f"chunk {chunk_idx+1} prompt")
        
        visual_prompt = None
        motion_prompt = None
        if response_data and "visual_prompt" in response_data and "motion_prompt" in response_data:
            visual_prompt = response_data["visual_prompt"]
            motion_prompt = response_data["motion_prompt"]
        
        if not visual_prompt or not motion_prompt:
            print(f"Using fallback prompts for chunk {chunk_idx + 1}")
            if chunk_prompts: # If there's a previous prompt, try to make it a continuation
                 visual_prompt = f"Continuing the scene: {original_scene_prompt[:50]}, focusing on '{current_narration_segment[:30]}...'"
                 motion_prompt = "Smooth continuation of the scene with subtle movement"
            else: # First chunk or isolated fallback
                 visual_prompt = f"{original_scene_prompt[:60]}, segment {chunk_idx+1} focusing on '{current_narration_segment[:30]}...'"
                 motion_prompt = "Gentle camera movement to establish the scene"
            # Ensure it's not overly long
            visual_prompt = (visual_prompt[:150] + '...') if len(visual_prompt) > 150 else visual_prompt
            motion_prompt = (motion_prompt[:150] + '...') if len(motion_prompt) > 150 else motion_prompt

        chunk_prompts.append((visual_prompt, motion_prompt))
        print(f"Generated prompts for chunk {chunk_idx + 1}:")
        print(f"  Visual: \"{visual_prompt[:60]}...\"")
        print(f"  Motion: \"{motion_prompt[:60]}...\"")

    return chunk_prompts