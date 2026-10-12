"""
Centralized logging configuration for TRAP attack method.
This module provides a single source of truth for all log paths.
"""
from pathlib import Path
import logging
import os

# Define the root directory for TRAP attack
OURS_ROOT = Path(__file__).parent.parent

# Define the centralized logs directory
LOGS_DIR = OURS_ROOT / 'logs'

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Define specific log file paths
LOG_FILES = {
    'main': LOGS_DIR / 'main_execution.log',
    'rl': LOGS_DIR / 'rl_attack.log',
    'rag': LOGS_DIR / 'rag_system.log',
    'training': LOGS_DIR / 'training.log',
    'dataset_attack': LOGS_DIR / 'dataset_attack.log',
}

# Define subdirectories
SUBDIRS = {
    'checkpoints': LOGS_DIR / 'checkpoints',
    'plots': LOGS_DIR / 'plots',
    'results': OURS_ROOT / 'results',
}

# Create subdirectories
for subdir in SUBDIRS.values():
    subdir.mkdir(parents=True, exist_ok=True)


def get_log_file(log_type='rl'):
    """
    Get the path to a specific log file.

    Args:
        log_type: Type of log file ('main', 'rl', 'rag', 'training')

    Returns:
        Path object for the log file
    """
    if log_type not in LOG_FILES:
        raise ValueError(f"Unknown log type: {log_type}. Available: {list(LOG_FILES.keys())}")
    return LOG_FILES[log_type]


def get_subdir(subdir_type):
    """
    Get the path to a specific subdirectory.

    Args:
        subdir_type: Type of subdirectory ('checkpoints', 'plots', 'results')

    Returns:
        Path object for the subdirectory
    """
    if subdir_type not in SUBDIRS:
        raise ValueError(f"Unknown subdir type: {subdir_type}. Available: {list(SUBDIRS.keys())}")
    return SUBDIRS[subdir_type]


def setup_logger(name, log_type='rl', level=logging.INFO):
    """
    Setup a logger with both file and console handlers.

    Args:
        name: Logger name (usually __name__)
        log_type: Type of log file to use
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # File handler
    log_file = get_log_file(log_type)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# For backward compatibility with common.py
STORE_FOLDER = str(LOGS_DIR)
