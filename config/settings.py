from typing import Dict, Any
from pathlib import Path
import multiprocessing

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# LLM Settings

LLM_CONFIG = {
    "model_id": "mlx-community/aya-expanse-8b-4bit",  # Changed from model_path
    "temperature": 0.4,
    "max_tokens": 8192,
    "top_p": 0.9,
    "repeat_penalty": 1.3,
}


EMBEDDING_CONFIG = {
    "model_path": str(BASE_DIR / "ai_models/bge-m3-korean-q4_k_m.gguf"),
    "chunk_size": 8192,
    "chunk_overlap": 200,
}

# Research Settings
RESEARCH_CONFIG = {
    "max_iterations": 5,
    "timeout": 300,
    "cache_results": True,
    "quality_threshold": 0.8,
}

# Quality Control Settings
QUALITY_CONFIG = {
    "min_confidence_score": 0.8,
    "validation_threshold": 0.7,
}

# Cache Settings
CACHE_CONFIG = {
    "enabled": True,
    "directory": str(BASE_DIR / "cache"),
    "ttl": 3600,  # Time to live in seconds
}
