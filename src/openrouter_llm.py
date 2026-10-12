"""
Shared OpenRouter LLM Class
This module provides a unified interface for OpenRouter API access
used by TRAP for model calls through OpenRouter.
"""

import os
import time
import json
import logging
import requests
from typing import Union, List, Dict

logger = logging.getLogger(__name__)


class OpenRouterAuthError(RuntimeError):
    """Raised when OpenRouter rejects the API key (HTTP 401).

    This is a non-recoverable account/key problem (e.g. expired or deleted key,
    'User not found'). It must propagate immediately rather than be retried or
    converted into a sentinel string, otherwise every downstream LLM/judge call
    silently produces garbage that gets scored 0 and cached.
    """


class OpenRouterLLM:
    """
    Universal OpenRouter LLM wrapper compatible with all attack methods
    """
    
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_RETRY_DELAY = 2
    DEFAULT_TIMEOUT = 60
    DEFAULT_BACKOFF_BASE = 1.5
    DEFAULT_BACKOFF_MAX = 12.0
    
    def __init__(self, 
                 model_name: str, 
                 api_key: str = None,
                 max_tokens: int = 500,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 max_retries: int = None,
                 retry_delay: int = None,
                 timeout: int = None,
                 backoff_base: float = None,
                 backoff_max: float = None,
                 skip_on_failure: bool = None,
                 skip_policy_errors: bool = None):
        """
        Initialize OpenRouter LLM
        
        Args:
            model_name: OpenRouter model identifier (e.g., 'openai/gpt-4o')
            api_key: OpenRouter API key (uses env var if not provided)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            backoff_base: Initial backoff seconds (exponential)
            backoff_max: Maximum backoff cap
            skip_on_failure: When True, return a sentinel string on transient failure instead of raising
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        # Allow env overrides to avoid long hangs
        env_max_retries = os.getenv("OPENROUTER_MAX_RETRIES")
        env_retry_delay = os.getenv("OPENROUTER_RETRY_DELAY")
        env_timeout = os.getenv("OPENROUTER_TIMEOUT")
        env_backoff_base = os.getenv("OPENROUTER_BACKOFF_BASE")
        env_backoff_max = os.getenv("OPENROUTER_BACKOFF_MAX")
        env_skip = os.getenv("OPENROUTER_SKIP_ON_FAILURE")
        env_skip_policy = os.getenv("OPENROUTER_SKIP_POLICY_ERRORS")
        
        self.max_retries = max_retries or int(env_max_retries) if env_max_retries else self.DEFAULT_MAX_RETRIES
        self.retry_delay = retry_delay or int(env_retry_delay) if env_retry_delay else self.DEFAULT_RETRY_DELAY
        self.timeout = timeout or int(env_timeout) if env_timeout else self.DEFAULT_TIMEOUT
        self.backoff_base = backoff_base or float(env_backoff_base) if env_backoff_base else self.DEFAULT_BACKOFF_BASE
        self.backoff_max = backoff_max or float(env_backoff_max) if env_backoff_max else self.DEFAULT_BACKOFF_MAX
        self.skip_on_failure = (
            skip_on_failure
            if skip_on_failure is not None
            else (env_skip.lower() == "true" if env_skip else False)
        )
        # When True, policy/auth errors (401/403) return a sentinel without retrying
        self.skip_policy_errors = (
            skip_policy_errors
            if skip_policy_errors is not None
            else (env_skip_policy.lower() == "true" if env_skip_policy else True)
        )
        
        # Get API key
        if api_key is None:
            api_key = os.environ.get('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError(
                    "OpenRouter API key not found. "
                    "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
                )
        
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/jailbreak-research",
            "X-Title": "Jailbreak Attack Research",
            "Content-Type": "application/json"
        }
        
        # For compatibility with different frameworks
        self.generation_config = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p
        }
        
        # For EasyJailbreak compatibility
        self.conversation = type('obj', (object,), {'messages': []})()
    
    def generate(self, 
                 prompt: Union[str, List[Dict]], 
                 max_tokens: int = None,
                 temperature: float = None,
                 top_p: float = None,
                 stop: List[str] = None,
                 **kwargs) -> str:
        """
        Generate response from OpenRouter API
        
        Args:
            prompt: Input prompt (string or list of message dicts)
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            top_p: Override default top_p
            stop: List of stop sequences
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        # Use provided parameters or defaults
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        
        # Guard against empty/whitespace-only prompts to avoid 400 errors
        if isinstance(prompt, str) and not prompt.strip():
            raise ValueError("Cannot call OpenRouter with an empty prompt")
        
        # Convert prompt to messages format
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            # Check if it's already in message format
            if prompt and isinstance(prompt[0], dict) and 'role' in prompt[0]:
                messages = prompt
            else:
                # Assume it's a list of strings (conversation)
                messages = [{"role": "user", "content": str(p)} for p in prompt]
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if stop:
            payload["stop"] = stop
        
        # DEBUG: Log payload before sending
        if os.getenv('DEBUG_OPENROUTER', '').lower() == 'true':
            print(f"\n[DEBUG_OPENROUTER] Sending payload:")
            print(f"  Model: {payload.get('model')}")
            print(f"  Temperature: {payload.get('temperature')}")
            print(f"  Max tokens: {payload.get('max_tokens')}")
            print(f"  Messages count: {len(payload.get('messages', []))}")
            if payload.get('messages'):
                for i, msg in enumerate(payload['messages']):
                    content_preview = msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content']
                    print(f"  Message {i} ({msg['role']}): {content_preview}")

        # Retry logic with better transient detection/backoff
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.OPENROUTER_API_URL,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    message = result['choices'][0]['message']
                    # Some OpenRouter backing providers (observed on
                    # openai/gpt-oss-120b) return content: null on a 200 --
                    # e.g. a refusal expressed only via the `refusal` field,
                    # or a reasoning-only turn. Calling .strip() on None
                    # raised an uncaught AttributeError that the generic
                    # `except Exception` below treated as transient, causing
                    # an infinite retry loop with ever-growing backoff
                    # instead of returning a legitimate empty/refusal result.
                    raw_content = message.get('content')
                    if raw_content is None:
                        raw_content = message.get('refusal') or ''
                    content = raw_content.strip()

                    # DEBUG: Log response
                    if os.getenv('DEBUG_OPENROUTER', '').lower() == 'true':
                        print(f"\n[DEBUG_OPENROUTER] Response received:")
                        print(f"  Raw content: '{content}'")
                        print(f"  Full result['choices'][0]: {result['choices'][0]}")

                    return content

                # 401 means a bad/expired/missing API key or account problem. This is
                # never recoverable per-prompt and silently returning a sentinel here
                # pollutes results and the result cache with garbage (score-0) runs.
                # Always fail-fast so the whole comparison stops with a clear message
                # instead of burning quota/time on a dead key.
                if response.status_code == 401:
                    error_summary = self._extract_error_summary(response)
                    raise OpenRouterAuthError(
                        f"OpenRouter authentication failed (401): {error_summary}. "
                        f"Check OPENROUTER_API_KEY in config.env."
                    )

                # 403 is typically per-prompt content/policy moderation, which is a
                # legitimate single-prompt skip rather than a fatal account error.
                if response.status_code == 403:
                    error_summary = self._extract_error_summary(response)
                    error_msg = f"OpenRouter policy error 403: {error_summary}"
                    if self.skip_policy_errors or self.skip_on_failure:
                        warning_msg = f"[POLICY_BLOCK] {error_msg}"
                        logger.warning(warning_msg)
                        return f"[OPENROUTER_POLICY_BLOCKED] {error_summary}"
                    last_error = RuntimeError(error_msg)
                    break

                # Retry on transient status codes
                if self._is_transient_status(response.status_code):
                    last_error = RuntimeError(
                        f"Transient status {response.status_code}: {response.text[:300]}"
                    )
                    self._wait_backoff(attempt, last_error)
                    continue

                # Other errors: treat as fatal
                raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text[:300]}")
                        
            except OpenRouterAuthError:
                # Never retry or skip a bad-key error — propagate immediately.
                raise
            except (
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.SSLError,
            ) as err:
                last_error = err
                self._wait_backoff(attempt, err)
                continue
            except Exception as err:
                last_error = err
                if attempt < self.max_retries:
                    self._wait_backoff(attempt, err)
                    continue
                raise

        if self.skip_on_failure:
            warning_msg = (
                f"[SKIP ITERATION] OpenRouter call failed after {self.max_retries} attempts "
                f"(model={self.model_name}): {last_error}"
            )
            logger.warning(warning_msg)
            return f"[OPENROUTER_SKIPPED_TRANSIENT] {warning_msg}"

        if last_error:
            # Propagate the underlying error instead of a generic wrapper when available
            if isinstance(last_error, RuntimeError):
                raise last_error
            raise RuntimeError(last_error)

        raise RuntimeError("OpenRouter call failed with an unknown error state")
    
    def batched_generate(self, 
                        prompts: List[Union[str, List[Dict]]], 
                        max_tokens: int = None,
                        temperature: float = None,
                        top_p: float = None,
                        extra_eos_tokens: List[str] = None,
                        max_n_tokens: int = None,
                        **kwargs) -> List[str]:
        """
        Generate responses for a batch of prompts
        
        Args:
            prompts: List of prompts (strings or message lists)
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            extra_eos_tokens: Additional stop sequences
            
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            if isinstance(prompt, str) and not prompt.strip():
                responses.append("Error: Empty prompt provided to OpenRouter")
                continue

            response = self.generate(
                prompt,
                max_tokens=max_n_tokens or max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=extra_eos_tokens
            )
            responses.append(response)
            time.sleep(1)  # Rate limiting
        return responses
    
    def batch_generate(self, conversations: List, **kwargs) -> List[str]:
        """
        Alias for batched_generate for compatibility
        """
        return self.batched_generate(conversations, **kwargs)
    
    def response(self, messages: Union[str, List[Dict]], **kwargs) -> str:
        """
        Get response (alias for generate, for compatibility)
        """
        return self.generate(messages, **kwargs)
    
    def __repr__(self):
        return f"OpenRouterLLM(model={self.model_name}, temp={self.temperature})"

    # Internal helpers
    def _is_transient_status(self, status_code: int) -> bool:
        """Treat 408/429 and 5xx as transient."""
        return status_code in (408, 429) or 500 <= status_code <= 599

    def _wait_backoff(self, attempt: int, err: Exception):
        """Sleep with exponential backoff bounded by backoff_max."""
        if attempt >= self.max_retries:
            return
        wait_time = min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))
        logger.warning(
            "OpenRouter attempt %d/%d failed (%s); backing off %.1fs",
            attempt,
            self.max_retries,
            err,
            wait_time,
        )
        time.sleep(wait_time)

    def _extract_error_summary(self, response) -> str:
        """
        Extract a short error summary from an OpenRouter HTTP response object.
        Handles both JSON bodies and raw text to keep logs readable.
        """
        # response may be either requests.Response or a mock; prefer .text
        body_text = getattr(response, "text", "") or ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                err = payload.get("error") or {}
                if isinstance(err, dict):
                    # Prefer provider message, fall back to raw provider error field
                    provider_msg = err.get("message") or err.get("error")
                    if provider_msg:
                        return str(provider_msg)
                # Fallback to top-level message
                top_level_msg = payload.get("message")
                if top_level_msg:
                    return str(top_level_msg)
        except Exception:
            pass
        # Default fallback to trimmed body
        return body_text[:300] if body_text else f"status={getattr(response, 'status_code', 'unknown')}"


def create_llm_from_config(config_key: str = 'TARGET', api_key: str = None) -> OpenRouterLLM:
    """
    Create an OpenRouterLLM instance from environment configuration
    
    Args:
        config_key: Which model to load ('ATTACK', 'TARGET', 'EVAL', 'JUDGE')
        api_key: Optional API key override
        
    Returns:
        Configured OpenRouterLLM instance
    """
    from dotenv import load_dotenv
    load_dotenv('config.env')
    
    model_name = os.getenv(f'{config_key}_MODEL')
    if not model_name:
        raise ValueError(f"{config_key}_MODEL not found in config.env")
    
    # Get appropriate parameters based on model type
    if config_key == 'ATTACK':
        max_tokens = int(os.getenv('ATTACK_MAX_TOKENS', '500'))
        temperature = float(os.getenv('ATTACK_TEMPERATURE', '1.0'))
        top_p = float(os.getenv('ATTACK_TOP_P', '0.9'))
    elif config_key == 'TARGET':
        max_tokens = int(os.getenv('TARGET_MAX_TOKENS', '150'))
        temperature = float(os.getenv('TARGET_TEMPERATURE', '0.0'))
        top_p = float(os.getenv('TARGET_TOP_P', '1.0'))
    elif config_key in ['EVAL', 'JUDGE']:
        max_tokens = int(os.getenv('JUDGE_MAX_TOKENS', '10'))
        temperature = float(os.getenv('JUDGE_TEMPERATURE', '0.0'))
        top_p = 1.0
    else:
        max_tokens = 500
        temperature = 1.0
        top_p = 0.9
    
    return OpenRouterLLM(
        model_name=model_name,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )


if __name__ == "__main__":
    # Test the LLM
    print("Testing OpenRouterLLM...")
    llm = create_llm_from_config('TARGET')
    print(f"Created: {llm}")
    
    test_prompt = "Say 'Hello, World!' and nothing else."
    print(f"\nTest prompt: {test_prompt}")
    response = llm.generate(test_prompt)
    print(f"Response: {response}")
