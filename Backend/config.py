import os
from pathlib import Path
from typing import Optional

# Handle pydantic version compatibility
try:
    from pydantic import BaseSettings
except ImportError:
    try:
        from pydantic.v1 import BaseSettings
    except ImportError:
        from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Model Paths
    MODELS_DIR: Path = Path("models")
    LLM_MODEL_PATH: Optional[str] = None
    
    # STT Settings
    WHISPER_MODEL_SIZE: str = "base"  # tiny, base, small, medium, large
    STT_LANGUAGE: str = "de"  # German by default
    
    # TTS Settings
    USE_COQUI_TTS: bool = True
    TTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    TTS_LANGUAGE: str = "de"
    
    # NLP Settings
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    VECTOR_STORE_PATH: Optional[str] = "data/vectorstore"
    KNOWLEDGE_BASE_PATH: Optional[str] = "data/knowledge_base.json"
    
    # LLM Settings
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 512
    LLM_CONTEXT_LENGTH: int = 2048
    
    # Performance Settings
    USE_GPU: bool = True
    MAX_CONCURRENT_REQUESTS: int = 5
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "logs/voicebot.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Model download URLs (for setup script)
MODEL_URLS = {
    "mistral-7b-instruct": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "size": "4.1GB"
    },
    "llama2-7b-chat": {
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        "filename": "llama-2-7b-chat.Q4_K_M.gguf", 
        "size": "4.1GB"
    }
}

def get_model_path(model_name: str) -> Path:
    """Get full path to model file"""
    model_info = MODEL_URLS.get(model_name)
    if not model_info:
        raise ValueError(f"Unknown model: {model_name}")
    
    return settings.MODELS_DIR / model_info["filename"]

def ensure_directories():
    """Create necessary directories"""
    directories = [
        settings.MODELS_DIR,
        Path("data"),
        Path("logs"),
        Path("temp")
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)

# Service configuration
STT_CONFIG = {
    "model_size": settings.WHISPER_MODEL_SIZE,
    "language": settings.STT_LANGUAGE
}

NLP_CONFIG = {
    "model_path": settings.LLM_MODEL_PATH or str(get_model_path("mistral-7b-instruct")),
    "embedding_model": settings.EMBEDDING_MODEL,
    "temperature": settings.LLM_TEMPERATURE,
    "max_tokens": settings.LLM_MAX_TOKENS,
    "context_length": settings.LLM_CONTEXT_LENGTH
}

TTS_CONFIG = {
    "use_coqui": settings.USE_COQUI_TTS,
    "model": settings.TTS_MODEL,
    "language": settings.TTS_LANGUAGE
}