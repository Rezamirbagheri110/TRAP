import os

# Model paths - use environment variables with fallbacks
VICUNA_PATH = os.getenv('VICUNA_PATH', './models/vicuna-13b-v1.5/')
LLAMA_PATH = os.getenv('LLAMA_PATH', './models/Llama-2-7b-chat-hf')

# API links - use environment variables with fallbacks
VICUNA_API_LINK = os.getenv('VICUNA_API_LINK', 'https://api.endpoints.anyscale.com/v1/chat/completions')
LLAMA_API_LINK = os.getenv('LLAMA_API_LINK', 'https://api.endpoints.anyscale.com/v1/chat/completions')


ATTACK_TEMP = 1
TARGET_TEMP = 0
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1

# Increase the above allow more streams in parallel
# Decrease it to reduce the memory requirement 
MAX_PARALLEL_STREAMS = 5
