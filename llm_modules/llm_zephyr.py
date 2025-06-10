# llm_modules/llm_zephyr.py
import torch
import json
import re
from typing import List, Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from base_modules import BaseLLM, BaseModuleConfig, ModuleCapabilities
from config_manager import ContentConfig, DEVICE, clear_vram_globally

class ZephyrLLMConfig(BaseModuleConfig):
    model_id: str = "HuggingFaceH4/zephyr-7b-beta"
    max_new_tokens_script: int = 2048 # Increased for new fields
    max_new_tokens_chunk_prompt: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

class ZephyrLLM(BaseLLM):
    Config = ZephyrLLMConfig

    @classmethod
    def get_capabilities(cls) -> ModuleCapabilities:
        return ModuleCapabilities(
            title="Zephyr 7B",
            vram_gb_min=8.0,
            ram_gb_min=16.0,
            # LLM-specific capabilities are not the main focus, so we use defaults.
        )
    
    def _load_model_and_tokenizer(self):
        if self.model is None or self.tokenizer is None:
            print(f"Loading LLM: {self.config.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id, torch_dtype=torch.float16
                ).to(DEVICE)
            except Exception as e:
                print(f"Failed to load LLM with device_map='auto' ({e}), trying with explicit device: {DEVICE}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id, torch_dtype=torch.float16
                ).to(DEVICE)
            print("LLM loaded.")

    def clear_vram(self):
        print("Clearing LLM VRAM...")
        models_to_clear = [m for m in [self.model] if m is not None]
        if models_to_clear: clear_vram_globally(*models_to_clear)
        self.model, self.tokenizer = None, None
        print("LLM VRAM cleared.")

    def _parse_llm_json_response(self, decoded_output: str, context: str = "script") -> Optional[Dict]:
        match = re.search(r'\{[\s\S]*\}', decoded_output)
        json_text = match.group(0) if match else decoded_output
        try:
            return json.loads(re.sub(r',(\s*[}\]])', r'\1', json_text))
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM JSON for {context}: {e}. Raw output:\n{decoded_output}")
            return None

    def generate_script(self, topic: str, content_config: ContentConfig) -> Dict[str, Any]:
        self._load_model_and_tokenizer()
        print(f"Generating script and context for topic: {topic}")
        
        system_prompt = (
            "You are an AI assistant creating content for a short video. "
            "First, describe the main subject and the general setting. "
            "Then, create the narration, visuals, and hashtags. "
            "Your response must be a single, valid JSON object with these exact keys: "
            "\"main_subject_description\", \"setting_description\", \"narration\", \"visuals\", \"hashtags\"."
        )
        
        user_prompt = f"""
        Create content for a short video about "{topic}".
        The total narration should be ~{content_config.target_video_length_hint}s, with {content_config.min_scenes} to {content_config.max_scenes} scenes.
        Each scene's narration should be ~{content_config.max_scene_narration_duration_hint}s.
        
        Return your response in this exact JSON format:
        {{
            "main_subject_description": "A detailed, consistent description of the main character or subject (e.g., 'Fluffy, a chubby but cute orange tabby cat with green eyes').",
            "setting_description": "A description of the primary environment (e.g., 'a cozy, sunlit living room with plush furniture').",
            "narration": [
                {{"scene": 1, "text": "First scene narration text.", "duration_estimate": {content_config.max_scene_narration_duration_hint}}}
            ],
            "visuals": [
                {{"scene": 1, "prompt": "Detailed visual prompt for scene 1, focusing on key elements, style, and mood."}}
            ],
            "hashtags": ["relevantTag1", "relevantTag2"]
        }}
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            input_ids=tokenized_chat, max_new_tokens=self.config.max_new_tokens_script, 
            do_sample=True, top_k=self.config.top_k, top_p=self.config.top_p, 
            temperature=self.config.temperature, pad_token_id=self.tokenizer.eos_token_id
        )
        decoded_output = self.tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)
        response_data = self._parse_llm_json_response(decoded_output, "script")

        if response_data and all(k in response_data for k in ["narration", "visuals", "main_subject_description"]):
            # Reformat for what ProjectManager expects
            return {
                "main_subject_description": response_data.get("main_subject_description"),
                "setting_description": response_data.get("setting_description"),
                "narration": sorted(response_data.get("narration", []), key=lambda x: x["scene"]),
                "visuals": [p["prompt"] for p in sorted(response_data.get("visuals", []), key=lambda x: x["scene"])],
                "hashtags": response_data.get("hashtags", [])
            }
        
        print("LLM script generation failed or gave invalid format. Using fallback.")
        return {
            "main_subject_description": topic, "setting_description": "a simple background",
            "narration": [{"text": f"An intro to {topic}.", "duration_estimate": 5.0}],
            "visuals": [f"Cinematic overview of {topic}."], "hashtags": [f"#{topic.replace(' ', '')}"]
        }

    def generate_chunk_visual_prompts(self, scene_narration: str, original_scene_prompt: str, num_chunks: int, content_config: ContentConfig, main_subject: str, setting: str) -> List[Tuple[str, str]]:
        self._load_model_and_tokenizer()
        chunk_prompts = []
        for chunk_idx in range(num_chunks):
            # The context is now a mandatory part of the prompt to the LLM
            system_prompt = (
                "You are an AI assistant. Your task is to generate a 'visual_prompt' and a 'motion_prompt' for a short video chunk. "
                "The prompts MUST incorporate the provided main subject and setting. Do NOT change the subject. "
                "Respond in this exact JSON format: {\"visual_prompt\": \"...\", \"motion_prompt\": \"...\"}"
            )
            user_prompt = f"""
            **Main Subject (MUST BE INCLUDED):** {main_subject}
            **Setting (MUST BE INCLUDED):** {setting}
            
            ---
            **Original Scene Goal:** "{original_scene_prompt}"
            **This Chunk's Narration:** "{scene_narration}"
            
            Based on ALL the information above, create a visual and motion prompt for chunk {chunk_idx + 1}/{num_chunks}.
            The visual prompt should be a specific, detailed moment consistent with the subject and setting.
            """
            
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                input_ids=tokenized_chat, max_new_tokens=self.config.max_new_tokens_chunk_prompt, 
                do_sample=True, temperature=self.config.temperature, pad_token_id=self.tokenizer.eos_token_id
            )
            decoded_output = self.tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)
            response_data = self._parse_llm_json_response(decoded_output, f"chunk {chunk_idx+1} prompt")

            visual_prompt, motion_prompt = None, None
            if isinstance(response_data, dict):
                visual_prompt = response_data.get("visual_prompt") if isinstance(response_data.get("visual_prompt"), str) else None
                motion_prompt = response_data.get("motion_prompt") if isinstance(response_data.get("motion_prompt"), str) else None

            if not visual_prompt or not motion_prompt:
                print(f"Using fallback prompts for chunk {chunk_idx + 1}")
                visual_prompt = f"{main_subject} in {setting}, {original_scene_prompt}"
                motion_prompt = "gentle camera movement"

            chunk_prompts.append((visual_prompt, motion_prompt))
            print(f"  > Chunk {chunk_idx+1} Visual: \"{visual_prompt[:80]}...\"")
            print(f"  > Chunk {chunk_idx+1} Motion: \"{motion_prompt[:80]}...\"")
        
        return chunk_prompts