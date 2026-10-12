"""
TRAP conversational model interfaces using OpenRouter
Enhanced with Action Tree and Prompt Template System integration
"""

import sys
from pathlib import Path
import json
import random

# Add root directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from utils import common
from models.language_models_openrouter import GPT
from config.config import ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P, MAX_PARALLEL_STREAMS

# Import enhanced components - use absolute imports ONLY
from core.action_tree import create_action_tree
from prompts.prompt_templates import create_prompt_system

def load_attack_and_target_models(args):
    """Load attack and target models using OpenRouter"""
    
    # Map model names to OpenRouter format if needed
    attack_model_name = args.attack_model
    target_model_name = args.target_model
    
    # Load attack model
    attack_llm = AttackLLM(
        model_name=attack_model_name,
        max_n_tokens=args.attack_max_n_tokens,
        max_n_attack_attempts=args.max_n_attack_attempts,
        temperature=ATTACK_TEMP,
        top_p=ATTACK_TOP_P,
        goal=args.goal,
        target_str=args.target_str
    )
    
    # Load target model
    target_llm = TargetLLM(
        model_name=target_model_name,
        max_n_tokens=args.target_max_n_tokens,
        temperature=TARGET_TEMP,
        top_p=TARGET_TOP_P
    )
    
    return attack_llm, target_llm


class AttackLLM():
    """Attack language model using OpenRouter with enhanced strategy selection"""
    
    def __init__(self, model_name: str, max_n_tokens: int, max_n_attack_attempts: int,
                 temperature: float, top_p: float, goal: str = "", target_str: str = ""):
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.goal = goal
        self.target_str = target_str
        
        # Create OpenRouter GPT model
        self.model = GPT(model_name)
        self.template = "gpt-4"  # Use GPT-4 template for all models
        
        # Initialize enhanced components
        self.action_tree = create_action_tree(tree_variant="full_active")
        self.prompt_system = create_prompt_system(strict_validation=False)
        
        # Track current strategy per conversation
        self.strategy_state = {}  # conv_id -> current_node_name
    
    def get_attack(self, convs_list, prompts_list):
        """Generate attack prompts using enhanced strategy selection"""
        assert len(convs_list) == len(prompts_list), "Mismatch between conversations and prompts"
        
        batchsize = len(convs_list)
        valid_outputs = [None] * batchsize
        
        # Generate attacks for each conversation
        for i, (conv, prompt) in enumerate(zip(convs_list, prompts_list)):
            try:
                # 1. Determine strategy based on conversation state
                conv_id = getattr(conv, 'self_id', f"conv_{i}")
                current_node = self.strategy_state.get(conv_id, "root")
                
                # Context for decision making
                context = {
                    "goal": self.goal,
                    "target_str": self.target_str,
                    "iteration": len(conv.messages) // 2,
                    "last_response": conv.messages[-1] if conv.messages else "",
                    "requires_jailbreak": True
                }
                
                # 2. Navigate action tree
                # If we're at root or previous attempt failed, find optimal path
                if current_node == "root":
                    optimal_path = self.action_tree.get_optimal_path(context)
                    next_node = optimal_path[-1]  # Use leaf node
                else:
                    # Try to transition or stay (simplified logic)
                    try:
                        transition_node = self.action_tree.transition(current_node, context)
                        next_node = transition_node.name if transition_node else current_node
                    except Exception:
                        # Fallback to optimal path from root
                        optimal_path = self.action_tree.get_optimal_path(context)
                        next_node = optimal_path[-1]
                
                self.strategy_state[conv_id] = next_node
                
                # 3. Generate prompt using template system
                generated_prompt = self.prompt_system.generate_prompt(
                    action_name=next_node,
                    goal=self.goal,
                    target_str=self.target_str,
                    use_variation=True,
                    optimize_selection=True
                )
                
                # 4. Construct attack dictionary
                attack_dict = {
                    "improvement": f"Using strategy: {next_node}",
                    "prompt": generated_prompt['user_prompt']
                }
                
                # Update conversation
                valid_outputs[i] = attack_dict
                
                # Add to conversation history
                # Note: We're bypassing the standard LLM generation here because we're using
                # the template system directly. In a full RL setup, we might ask the LLM
                # to fill in the template, but here the template system does the heavy lifting.
                
                # However, to maintain iterative prompt refinement,
                # we should ideally let the LLM refine the prompt.
                # For now, we'll use the template output directly as it's high quality.
                
                json_str = json.dumps(attack_dict)
                conv.update_last_message(json_str)
                
            except Exception as e:
                print(f"[ERROR] Strategy generation failed for stream {i}: {e}")
                # Fallback to standard generation if strategy fails
                # Return the objective if strategy generation fails.
                valid_outputs[i] = {
                    "improvement": "Fallback due to error",
                    "prompt": self.goal  # Simple fallback
                }
        
        return valid_outputs


class TargetLLM():
    """Target language model using OpenRouter"""
    
    def __init__(self, model_name: str, max_n_tokens: int, temperature: float, 
                 top_p: float, preloaded_model=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        
        # Create OpenRouter GPT model
        self.model = GPT(model_name)
        self.template = "gpt-4"
    
    def get_response(self, prompts_list):
        """Get responses from target model"""
        batchsize = len(prompts_list)
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            full_prompts.append(conv.to_openai_api_messages())
        
        # Query in batches
        outputs_list = []
        for left in range(0, len(full_prompts), MAX_PARALLEL_STREAMS):
            right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts))
            
            if right == left:
                continue
            
            print(f'\tQuerying target with {right-left} prompts')
            
            outputs_list.extend(
                self.model.batched_generate(
                    full_prompts[left:right],
                    max_n_tokens=self.max_n_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            )
        
        return outputs_list
