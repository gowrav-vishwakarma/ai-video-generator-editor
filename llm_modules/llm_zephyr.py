# llm_modules/llm_zephyr.py
import torch
import json
import re
from typing import List, Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from base_modules import BaseLLM, BaseModuleConfig
from config_manager import ContentConfig, DEVICE, clear_vram_globally

class ZephyrLLMConfig(BaseModuleConfig):
    model_id: str = "HuggingFaceH4/zephyr-7b-beta"
    max_new_tokens_script: int = 1536
    max_new_tokens_chunk_prompt: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

class ZephyrLLM(BaseLLM):
    Config = ZephyrLLMConfig
    
    def _load_model_and_tokenizer(self):
        if self.model is None or self.tokenizer is None:
            print(f"Loading LLM: {self.config.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id, torch_dtype=torch.float16, device_map="auto"
                )
            except Exception as e:
                print(f"Failed to load LLM with device_map='auto' ({e}), trying with explicit device: {DEVICE}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id, torch_dtype=torch.float16
                ).to(DEVICE)
            print("LLM loaded.")

    def clear_vram(self):
        print("Clearing LLM VRAM...")
        models_to_clear = [m for m in [self.model] if m is not None]
        if models_to_clear:
            clear_vram_globally(*models_to_clear)
        self.model = None
        self.tokenizer = None
        print("LLM VRAM cleared.")

    def _parse_llm_json_response(self, decoded_output: str, context: str = "script") -> Optional[Dict]:
        match = re.search(r'\{[\s\S]*\}', decoded_output)
        if match:
            json_text_to_parse = match.group(0)
        else:
            json_text_to_parse = decoded_output
            print(f"Warning: Could not extract a clear JSON block for {context}, attempting to parse full output.")

        try:
            json_text_to_parse = re.sub(r',(\s*[}\]])', r'\1', json_text_to_parse) # Trailing commas
            return json.loads(json_text_to_parse)
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM JSON for {context}: {e}. Raw output was:\n{decoded_output}")
            return None

    def generate_script(
        self, topic: str, content_config: ContentConfig
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        self._load_model_and_tokenizer()
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

        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        generation_kwargs = {
            "input_ids": tokenized_chat, "max_new_tokens": self.config.max_new_tokens_script, 
            "do_sample": True, "top_k": self.config.top_k, "top_p": self.config.top_p, 
            "temperature": self.config.temperature, "pad_token_id": self.tokenizer.eos_token_id
        }

        outputs = self.model.generate(**generation_kwargs)
        decoded_output = self.tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)
        response_data = self._parse_llm_json_response(decoded_output, "script")

        if response_data:
            try:
                narration_data = sorted(response_data.get("narration", []), key=lambda x: x["scene"])
                visuals_data = sorted(response_data.get("visuals", []), key=lambda x: x["scene"])
                narration_scenes = [{"text": s["text"], "duration_estimate": float(s.get("duration_estimate", content_config.max_scene_narration_duration_hint))} for s in narration_data]
                visual_prompts = [s["prompt"] for s in visuals_data]
                hashtags = response_data.get("hashtags", [])
                return narration_scenes, visual_prompts, hashtags
            except (ValueError, KeyError) as e:
                print(f"Error processing parsed LLM script data: {e}. Using fallback.")
        
        fallback_duration = content_config.max_scene_narration_duration_hint
        narration_scenes = [{"text": f"An intro to {topic}.", "duration_estimate": fallback_duration}]
        visual_prompts = [f"Cinematic overview of {topic}."]
        hashtags = [f"#{topic.replace(' ', '')}"]
        return narration_scenes, visual_prompts, hashtags

    def generate_chunk_visual_prompts(
        self, scene_narration: str, original_scene_prompt: str, num_chunks: int, content_config: ContentConfig
    ) -> List[Tuple[str, str]]:
        self._load_model_and_tokenizer()
        print(f"Generating {num_chunks} chunk-specific visual prompts for scene...")
        
        chunk_prompts = []
        for chunk_idx in range(num_chunks):
            narration_words = scene_narration.split()
            words_per_chunk = len(narration_words) / num_chunks
            start_idx = int(chunk_idx * words_per_chunk)
            end_idx = int((chunk_idx + 1) * words_per_chunk)
            current_narration_segment = " ".join(narration_words[start_idx:end_idx])
            previous_prompt_context = f'Previous chunk showed: "{chunk_prompts[-1][0]}"' if chunk_prompts else 'This is the first chunk.'

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant creating concise visual and motion prompts for short video chunks. "
                        "Keep prompts under 77 tokens. Ensure visual continuity. "
                        "Respond in this exact JSON format: {\"visual_prompt\": \"your_visual_prompt\", \"motion_prompt\": \"your_motion_prompt\"}"
                    )
                },
                {
                    "role": "user",
                    "content": f"""
                    Original scene visual prompt: "{original_scene_prompt}"
                    This chunk ({chunk_idx + 1}/{num_chunks}) covers narration: "{current_narration_segment or 'a continuation of the scene'}"
                    {previous_prompt_context}
                    Generate a visual and motion prompt for THIS chunk only.
                    """
                }
            ]

            tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
            generation_kwargs = {
                "input_ids": tokenized_chat, "max_new_tokens": self.config.max_new_tokens_chunk_prompt, 
                "do_sample": True, "top_k": self.config.top_k, "top_p": self.config.top_p, 
                "temperature": self.config.temperature, "pad_token_id": self.tokenizer.eos_token_id
            }
            outputs = self.model.generate(**generation_kwargs)
            decoded_output = self.tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)
            response_data = self._parse_llm_json_response(decoded_output, f"chunk {chunk_idx+1} prompt")
            
            # --- START OF THE FIX ---
            visual_prompt = None
            motion_prompt = None

            # Check if response_data is a dictionary and has the required keys.
            if isinstance(response_data, dict):
                visual_prompt = response_data.get("visual_prompt")
                motion_prompt = response_data.get("motion_prompt")
                
                # Further validation: ensure prompts are strings, not complex objects
                if not isinstance(visual_prompt, str):
                    print(f"Warning: 'visual_prompt' was not a string for chunk {chunk_idx+1}. Using fallback.")
                    visual_prompt = None # Invalidate it to trigger fallback
                if not isinstance(motion_prompt, str):
                    print(f"Warning: 'motion_prompt' was not a string for chunk {chunk_idx+1}. Using fallback.")
                    motion_prompt = None # Invalidate it to trigger fallback

            # If any part of the validation failed, use a robust fallback.
            if not visual_prompt or not motion_prompt:
                print(f"Using fallback prompts for chunk {chunk_idx + 1}")
                if chunk_prompts: # If there's a previous prompt, try to make it a continuation
                     visual_prompt = f"Continuing the scene: {original_scene_prompt[:100]}, focusing on '{current_narration_segment[:50]}...'"
                     motion_prompt = "Smooth continuation of the scene with subtle movement"
                else: # First chunk or isolated fallback
                     visual_prompt = f"{original_scene_prompt[:120]}, segment {chunk_idx+1} focusing on '{current_narration_segment[:50]}...'"
                     motion_prompt = "Gentle camera movement to establish the scene"
            # --- END OF THE FIX ---

            chunk_prompts.append((visual_prompt, motion_prompt))
            print(f"  > Chunk {chunk_idx+1} Visual: \"{visual_prompt[:80]}...\"")
            print(f"  > Chunk {chunk_idx+1} Motion: \"{motion_prompt[:80]}...\"")
        
        return chunk_prompts