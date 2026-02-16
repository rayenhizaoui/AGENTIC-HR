"""
Shared Utilities - Common tools and helpers for all agents.

This module contains generic utilities used across the system:
- Logging configuration
- Common helpers
- Environment configuration
"""

import logging
import os
from functools import lru_cache
from typing import Optional
from pathlib import Path


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__).
        level: Logging level.
        log_file: Optional file path for logging.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Default logger for the package
logger = setup_logger("hr_platform")


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

@lru_cache()
def get_env_config() -> dict:
    """
    Load environment configuration.
    
    Returns:
        Dictionary with configuration values.
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    return {
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "mistral"),
        "redis_url": os.getenv("REDIS_URL", ""),
        "chroma_persist_dir": os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
        "templates_dir": os.getenv("TEMPLATES_DIR", "./data/templates"),
        "cv_upload_dir": os.getenv("CV_UPLOAD_DIR", "./data/uploads"),
        "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
    }


def ensure_directories():
    """Create necessary data directories if they don't exist."""
    config = get_env_config()
    
    directories = [
        config["chroma_persist_dir"],
        config["templates_dir"],
        config["cv_upload_dir"],
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")


# ============================================================
# COMMON HELPERS
# ============================================================

def safe_get(dictionary: dict, *keys, default=None):
    """
    Safely get nested dictionary values.
    
    Args:
        dictionary: The dictionary to search.
        *keys: Keys to traverse.
        default: Default value if not found.
    
    Returns:
        The value or default.
    
    Example:
        safe_get(data, "user", "profile", "name", default="Unknown")
    """
    result = dictionary
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, default)
        else:
            return default
    return result


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate.
        max_length: Maximum length.
        suffix: Suffix to add if truncated.
    
    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_skills_list(skills: list[str], max_display: int = 10) -> str:
    """
    Format a list of skills for display.
    
    Args:
        skills: List of skill strings.
        max_display: Maximum skills to show.
    
    Returns:
        Formatted skills string.
    """
    if not skills:
        return "No skills listed"
    
    displayed = skills[:max_display]
    remaining = len(skills) - max_display
    
    result = ", ".join(displayed)
    if remaining > 0:
        result += f" (+{remaining} more)"
    
    return result


def normalize_skill(skill: str) -> str:
    """
    Normalize a skill string for comparison.
    
    Args:
        skill: Raw skill string.
    
    Returns:
        Normalized skill string.
    """
    return skill.lower().strip().replace("-", " ").replace("_", " ")


# ============================================================
# STATE HELPERS
# ============================================================

def create_initial_state(user_message: str, job_context: Optional[dict] = None) -> dict:
    """
    Create an initial agent state for a new conversation.
    
    Args:
        user_message: The user's input message.
        job_context: Optional initial context.
    
    Returns:
        Initial state dictionary.
    """
    from langchain_core.messages import HumanMessage
    
    return {
        "messages": [HumanMessage(content=user_message)],
        "next": "",
        "job_context": job_context or {}
    }


def extract_last_message(state: dict) -> Optional[str]:
    """
    Extract the content of the last message from state.
    
    Args:
        state: Agent state dictionary.
    
    Returns:
        Last message content or None.
    """
    messages = state.get("messages", [])
    if not messages:
        return None
    
    last_msg = messages[-1]
    return last_msg.content if hasattr(last_msg, 'content') else str(last_msg)


# ============================================================
# ERROR HANDLING
# ============================================================

class HRPlatformError(Exception):
    """Base exception for HR Platform errors."""
    pass


class CVParsingError(HRPlatformError):
    """Error during CV parsing."""
    pass


class TemplateNotFoundError(HRPlatformError):
    """Error when template is not found."""
    pass


class SkillExtractionError(HRPlatformError):
    """Error during skill extraction."""
    pass
