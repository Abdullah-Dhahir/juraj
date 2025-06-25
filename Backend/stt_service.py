import whisper
import tempfile
import os
import asyncio
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class STTService:
    """Speech-to-Text service using OpenAI Whisper"""
    
    def __init__(self, model_size="base"):
        """
        Initialize STT service with Whisper model
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model_name = f"whisper-{model_size}"
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"🔄 Loading Whisper model ({self.model_size})...")
            self.model = whisper.load_model(self.model_size)
            logger.info(f"✅ Whisper model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            raise
    
    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Transcribed text
        """
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._transcribe_file, 
                temp_file_path
            )
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            # Clean up temp file if it exists
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise
    
    def _transcribe_file(self, audio_path: str) -> str:
        """
        Internal method to transcribe audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language="de",  # German first, but Whisper can auto-detect
                task="transcribe"
            )
            
            # Extract text from result
            text = result["text"].strip()
            
            # Log detected language
            detected_language = result.get("language", "unknown")
            logger.info(f"🌍 Detected language: {detected_language}")
            
            return text
            
        except Exception as e:
            logger.error(f"❌ Whisper transcription failed: {e}")
            raise
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        return {
            "de": "German",
            "en": "English",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian",
            "nl": "Dutch",
            "pl": "Polish",
            "pt": "Portuguese",
            "ru": "Russian"
        }
    
    def get_model_info(self):
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_size": self.model_size,
            "supported_languages": list(self.get_supported_languages().keys()),
            "status": "loaded" if self.model else "not_loaded"
        }