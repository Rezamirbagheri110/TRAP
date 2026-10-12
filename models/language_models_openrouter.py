"""
TRAP language model interfaces using OpenRouter
This module uses the shared OpenRouter LLM client
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

# Dynamic import for OpenRouterLLM - supports running from different directories
def _get_openrouter_llm():
    """Dynamically import OpenRouterLLM from the project's src directory."""
    # Try different import strategies
    try:
        from src.openrouter_llm import OpenRouterLLM
        return OpenRouterLLM
    except ImportError:
        pass
    
    # Try adding project root to path
    current_file = Path(__file__).resolve()
    # Navigate up: models -> TRAP -> attacks -> project_root
    project_root = current_file.parent.parent
    src_path = project_root / 'src'
    
    if src_path.exists() and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        from src.openrouter_llm import OpenRouterLLM
        return OpenRouterLLM
    except ImportError:
        pass
    
    # Final fallback: inline implementation
    return _create_inline_openrouter_llm()

def _create_inline_openrouter_llm():
    """Create an inline OpenRouterLLM class as fallback."""
    import requests
    import json
    import time
    
    class OpenRouterLLM:
        """Fallback OpenRouter LLM wrapper."""
        
        OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
        DEFAULT_MAX_RETRIES = 5
        DEFAULT_RETRY_DELAY = 2
        DEFAULT_TIMEOUT = 60
        
        def __init__(self, model_name: str, api_key: str = None,
                     max_tokens: int = 500, temperature: float = 1.2,
                     top_p: float = 0.9, max_retries: int = None,
                     retry_delay: int = None, timeout: int = None, **kwargs):
            self.model_name = model_name
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.top_p = top_p
            # Env overrides to avoid long hangs
            env_max_retries = os.getenv("OPENROUTER_MAX_RETRIES")
            env_retry_delay = os.getenv("OPENROUTER_RETRY_DELAY")
            env_timeout = os.getenv("OPENROUTER_TIMEOUT")
            self.max_retries = max_retries or int(env_max_retries) if env_max_retries else self.DEFAULT_MAX_RETRIES
            self.retry_delay = retry_delay or int(env_retry_delay) if env_retry_delay else self.DEFAULT_RETRY_DELAY
            self.timeout = timeout or int(env_timeout) if env_timeout else self.DEFAULT_TIMEOUT
            
            if api_key is None:
                api_key = os.environ.get('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found")
            
            self.api_key = api_key
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/jailbreak-research",
                "X-Title": "Jailbreak Attack Research",
                "Content-Type": "application/json"
            }
        
        def generate(self, prompt, max_tokens: int = None, temperature: float = None,
                     top_p: float = None, **kwargs) -> str:
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature if temperature is not None else self.temperature
            top_p = top_p if top_p is not None else self.top_p
            
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        self.OPENROUTER_API_URL,
                        headers=self.headers,
                        data=json.dumps(payload),
                        timeout=self.timeout
                    )
                    if response.status_code == 200:
                        result = response.json()
                        return result['choices'][0]['message']['content'].strip()
                    if response.status_code == 429:
                        wait_time = self.retry_delay * (attempt + 1)
                        time.sleep(wait_time)
                        continue
                    time.sleep(self.retry_delay)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(self.retry_delay)
            return "ERROR: Failed to get response"
        
        def batched_generate(self, prompts, max_tokens=None, temperature=None,
                            top_p=None, **kwargs):
            return [self.generate(p, max_tokens, temperature, top_p) for p in prompts]
    
    return OpenRouterLLM

# Get the OpenRouterLLM class
OpenRouterLLM = _get_openrouter_llm()

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name
    
    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError

class GPT(LanguageModel):
    """OpenRouter-based GPT wrapper for TRAP"""
    
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 20
    API_TIMEOUT = 20
    
    def __init__(self, model_name):
        super().__init__(model_name)
        # Map old model names to OpenRouter format
        model_map = {
            'gpt-3.5-turbo': 'openai/gpt-3.5-turbo',
            'gpt-4': 'openai/gpt-4',
            'gpt-4-turbo': 'openai/gpt-4-turbo',
            'gpt-4-1106-preview': 'openai/gpt-4-turbo'
        }
        
        # Only add openai/ prefix if model doesn't already have a provider prefix
        if '/' in model_name:
            openrouter_model_name = model_name
        else:
            openrouter_model_name = model_map.get(model_name, f'openai/{model_name}')
        
        # Get API key from root config
        api_key = os.getenv('OPENROUTER_API_KEY')
        self.llm = OpenRouterLLM(
            model_name=openrouter_model_name,
            api_key=api_key,
            max_retries=self.API_MAX_RETRY
        )
    
    def generate(self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float):
        """Generate single response"""
        return self.llm.generate(conv, max_tokens=max_n_tokens, temperature=temperature, top_p=top_p)
    
    def batched_generate(self, convs_list: List[List[Dict]], max_n_tokens: int, 
                        temperature: float, top_p: float = 1.0):
        """Generate batch of responses"""
        return self.llm.batched_generate(convs_list, max_tokens=max_n_tokens, 
                                        temperature=temperature, top_p=top_p)

