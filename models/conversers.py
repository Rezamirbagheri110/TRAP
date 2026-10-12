
import os
import sys
import logging
from typing import Optional, Tuple, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import with error handling - use absolute imports to avoid "attempted relative import beyond top-level package"
try:
    from utils import common
    from models.language_models_openrouter import GPT
    # Keep these for backward compatibility (they're in language_models.py)
    try:
        from models.language_models import HuggingFace
    except ImportError:
        HuggingFace = None
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    # Create dummy classes to prevent crashes
    class GPT:
        def __init__(self, model_name): pass
    class HuggingFace:
        def __init__(self, model_name, model, tokenizer): pass
    class common:
        @staticmethod
        def conv_template(template): pass
        @staticmethod
        def extract_json(text): return None, ""

# Note: PaLM, APIModelLlama7B, APIModelVicuna13B, GeminiPro are disabled/not used
# They were in language_models.py but are commented out or disabled

# Import configuration with error handling - use absolute imports ONLY
try:
    from config.config import VICUNA_PATH, LLAMA_PATH, ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P, MAX_PARALLEL_STREAMS
except ImportError:
    # Fallback configuration
    VICUNA_PATH = os.getenv('VICUNA_PATH', './models/vicuna-13b-v1.5/')
    LLAMA_PATH = os.getenv('LLAMA_PATH', './models/Llama-2-7b-chat-hf')
    ATTACK_TEMP = 1
    TARGET_TEMP = 0
    ATTACK_TOP_P = 0.9
    TARGET_TOP_P = 1
    MAX_PARALLEL_STREAMS = 5
    logging.warning("Using fallback configuration values")

def load_target_model(args):
    target_llm = TargetLLM(model_name = args.target_model, 
                        max_n_tokens = args.target_max_n_tokens,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        )
    return target_llm

def load_attack_and_target_models(args):
    # Load attack model and tokenizer
    attack_llm = AttackLLM(model_name = args.attack_model, 
                        max_n_tokens = args.attack_max_n_tokens, 
                        max_n_attack_attempts = args.max_n_attack_attempts, 
                        temperature = ATTACK_TEMP, # init to 1
                        top_p = ATTACK_TOP_P, # init to 0.9
                        )
    preloaded_model = None
    if args.attack_model == args.target_model:
        print("Using same attack and target model. Using previously loaded model.")
        preloaded_model = attack_llm.model
    target_llm = TargetLLM(model_name = args.target_model, 
                        max_n_tokens = args.target_max_n_tokens,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        preloaded_model = preloaded_model,
                        )
    return attack_llm, target_llm

class AttackLLM():
    """
        Base class for attacker language models.
        Generates attacks for conversations using a language model. 
        The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                max_n_attack_attempts: int, 
                temperature: float,
                top_p: float):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)
        
        if "vicuna" in model_name or "llama" in model_name:
            if "api-model" not in model_name:
                self.model.extend_eos_tokens()

    def get_attack(self, convs_list, prompts_list):
        """
            Generates responses for a batch of conversations and prompts using a language model. 
            Only valid outputs in proper JSON format are returned. If an output isn't generated 
            successfully after max_n_attack_attempts, it's returned as None.
            
            Parameters:
            - convs_list: List of conversation objects.
            - prompts_list: List of prompts corresponding to each conversation.
            
            Returns:
            - List of generated outputs (dictionaries) or None for failed generations.
        """
        
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \"""" 

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])
            
        for _ in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs 
            #     Query the attack LLM in batched-queries 
            #       with at most MAX_PARALLEL_STREAMS-many queries at a time
            outputs_list = []
            for left in range(0, len(full_prompts_subset), MAX_PARALLEL_STREAMS):
                right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts_subset)) 
                
                if right == left:
                    continue 

                print(f'\tQuerying attacker with {len(full_prompts_subset[left:right])} prompts')
                
                outputs_list.extend(
                                    self.model.batched_generate(full_prompts_subset[left:right],
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p
                                                    )
                )
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                
                if "gpt" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output)
                
                # Now extract_json always returns a valid dict (with fallback if needed)
                # Check if it's a real success or a fallback
                if attack_dict is not None:
                    # Check if this is a fallback response (generic prompt or specific keywords indicate failure)
                    is_fallback = ("Please provide the requested information" in attack_dict.get("prompt", "") or
                                 "refused" in attack_dict.get("improvement", "").lower() or
                                 "failed" in attack_dict.get("improvement", "").lower() or
                                 "missing" in attack_dict.get("improvement", "").lower())
                    
                    if not is_fallback:
                        # Real success - use it
                        valid_outputs[orig_index] = attack_dict
                        # Update the conversation with valid generation
                        convs_list[orig_index].update_last_message(json_str)
                    else:
                        # Fallback response - try to regenerate
                        print(f"[WARNING] Fallback response detected for index {orig_index}: {attack_dict.get('improvement', '')[:50]}...")
                        new_indices_to_regenerate.append(orig_index)
                else:
                    # This shouldn't happen anymore, but keep for safety
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate 
            
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break
        
        # CRITICAL FIX: Use fallback responses for remaining failures instead of crashing
        # This allows the attack to continue even if some streams fail
        for i, output in enumerate(valid_outputs):
            if output is None:
                # Provide a fallback response with non-empty prompt to avoid API errors
                print(f"[WARNING] Using fallback for conversation {i} after {self.max_n_attack_attempts} attempts")
                fallback_message = "Please provide the requested information."
                valid_outputs[i] = {
                    "improvement": f"Failed to generate valid adversarial content after {self.max_n_attack_attempts} attempts",
                    "prompt": fallback_message
                }
                # CRITICAL FIX: Update conversation with fallback to maintain consistency
                convs_list[i].update_last_message(fallback_message)
        
        # Log summary of any failures (check for generic fallback prompts)
        failed_count = sum(1 for output in valid_outputs if "Please provide the requested information" in output.get("prompt", ""))
        if failed_count > 0:
            print(f"[WARNING] Using fallback responses for {failed_count}/{len(valid_outputs)} conversations")
            print(f"[TIP] Consider increasing max_n_attack_attempts or using a stronger attack model")
        
        return valid_outputs

class TargetLLM():
    """
        Base class for target language models.
        Generates responses for prompts using a language model. 
        The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
            model_name: str, 
            max_n_tokens: int, 
            temperature: float,
            top_p: float,
            preloaded_model: object = None):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_response(self, prompts_list):
        batchsize = len(prompts_list)
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                # OpenAI does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None) 
                full_prompts.append(conv.get_prompt())

        # Query the attack LLM in batched-queries with at most MAX_PARALLEL_STREAMS-many queries at a time
        outputs_list = []
        for left in range(0, len(full_prompts), MAX_PARALLEL_STREAMS):
            right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts))
            
            if right == left:
                continue 

            print(f'\tQuerying target with {len(full_prompts[left:right])} prompts')


            outputs_list.extend(
                                self.model.batched_generate(full_prompts[left:right], 
                                                            max_n_tokens = self.max_n_tokens,  
                                                            temperature = self.temperature,
                                                            top_p = self.top_p
                                                        )
            )
        return outputs_list



def load_indiv_model(model_name: str) -> Tuple[Any, str]:
    """
    Load individual model with enhanced error handling and validation.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model_instance, template)
        
    Raises:
        ValueError: If model_name is not supported
        FileNotFoundError: If local model path doesn't exist
        ImportError: If required dependencies are missing
    """
    try:
        model_path, template = get_model_path_and_template(model_name)
        
        # Set global model name for common module
        if hasattr(common, 'MODEL_NAME'):
            common.MODEL_NAME = model_name
        
        # API-based models
        if model_name in ["gpt-3.5-turbo", "gpt-4", 'gpt-4-1106-preview', 'gpt-4-turbo']:
            lm = GPT(model_name)
        elif model_name == "palm-2":
            # PaLM not supported - use GPT as fallback
            logging.warning(f"PaLM not supported, using GPT as fallback for {model_name}")
            lm = GPT('openai/gpt-3.5-turbo')
        elif model_name == "gemini-pro":
            # GeminiPro not supported - use GPT as fallback
            logging.warning(f"GeminiPro not supported, using GPT as fallback for {model_name}")
            lm = GPT('openai/gpt-3.5-turbo')
        elif model_name == 'llama-2-api-model':
            # API models not supported - use GPT as fallback
            logging.warning(f"API model not supported, using GPT as fallback for {model_name}")
            lm = GPT('openai/gpt-3.5-turbo')
        elif model_name == 'vicuna-api-model':
            # API models not supported - use GPT as fallback
            logging.warning(f"API model not supported, using GPT as fallback for {model_name}")
            lm = GPT('openai/gpt-3.5-turbo')
        else:
            # Local model loading with enhanced error handling
            if not model_path or model_path.startswith("..."):
                raise ValueError(f"Model path not configured for {model_name}. Please set environment variables or update config.")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                ).eval()

                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    use_fast=False
                ) 

                # Configure tokenizer based on model type
                if 'llama-2' in model_path.lower():
                    tokenizer.pad_token = tokenizer.unk_token
                    tokenizer.padding_side = 'left'
                elif 'vicuna' in model_path.lower():
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.padding_side = 'left'
                elif not tokenizer.pad_token:
                    tokenizer.pad_token = tokenizer.eos_token

                lm = HuggingFace(model_name, model, tokenizer)
                
            except Exception as e:
                raise ImportError(f"Failed to load local model {model_name}: {str(e)}")
        
        logging.info(f"Successfully loaded model: {model_name}")
        return lm, template
        
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {str(e)}")
        raise

def get_model_path_and_template(model_name: str) -> Tuple[Optional[str], str]:
    """
    Get model path and template with enhanced error handling.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Tuple of (model_path, template)
        
    Raises:
        ValueError: If model_name is not supported
    """
    full_model_dict = {
        "gpt-4-1106-preview": {
            "path": "gpt-4-1106-preview",
            "template": "gpt-4-1106-preview"
        },
        "gpt-4-turbo": {
            "path": "gpt-4-1106-preview",
            "template": "gpt-4-1106-preview"
        },
        "gpt-4": {
            "path": "gpt-4",
            "template": "gpt-4"
        },
        "gpt-3.5-turbo": {
            "path": "gpt-3.5-turbo",
            "template": "gpt-3.5-turbo"
        },
        "vicuna": {
            "path": VICUNA_PATH,
            "template": "vicuna_v1.1"
        },
        "vicuna-api-model": {
            "path": None,
            "template": "vicuna_v1.1"
        },
        "llama-2": {
            "path": LLAMA_PATH,
            "template": "llama-2"
        },
        "llama-2-api-model": {
            "path": None,
            "template": "llama-2-7b"
        },
        "palm-2": {
            "path": "palm-2",
            "template": "palm-2"
        },
        "gemini-pro": {
            "path": "gemini-pro",
            "template": "gemini-pro"
        }
    }
    
    if model_name not in full_model_dict:
        available_models = list(full_model_dict.keys())
        raise ValueError(f"Unsupported model '{model_name}'. Available models: {available_models}")
    
    model_config = full_model_dict[model_name]
    path = model_config["path"]
    template = model_config["template"]
    
    # Validate local model paths
    if path and not path.startswith(("http", "gpt-", "palm-", "gemini-")):
        if path.startswith("..."):
            raise ValueError(f"Model path not configured for {model_name}. Please set the appropriate environment variable.")
        if not os.path.exists(path):
            logging.warning(f"Model path does not exist: {path}")
    
    return path, template



    

