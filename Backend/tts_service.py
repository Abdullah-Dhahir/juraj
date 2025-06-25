import asyncio
import logging
import tempfile
import os
import io
from pathlib import Path
from typing import Optional
import langdetect

# TTS imports - using pyttsx3 as fallback, Coqui TTS as primary
try:
    import torch
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    logging.warning("⚠️ Coqui TTS not available, using fallback TTS")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning("⚠️ pyttsx3 not available")

logger = logging.getLogger(__name__)

class TTSService:
    """Text-to-Speech service with Coqui TTS and fallback options"""
    
    def __init__(self, use_coqui: bool = True):
        """
        Initialize TTS service
        
        Args:
            use_coqui: Whether to use Coqui TTS (if available)
        """
        self.model_name = "tts_service"
        self.tts_engine = None
        self.coqui_tts = None
        self.use_coqui = use_coqui and COQUI_AVAILABLE
        
        # Initialize TTS engines
        self._initialize_tts()
    
    def _initialize_tts(self):
        """Initialize TTS engines"""
        try:
            if self.use_coqui:
                self._init_coqui_tts()
            else:
                self._init_fallback_tts()
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize primary TTS: {e}")
            logger.info("🔄 Falling back to alternative TTS...")
            self._init_fallback_tts()
    
    def _init_coqui_tts(self):
        """Initialize Coqui TTS"""
        try:
            logger.info("🔄 Loading Coqui TTS...")
            
            # Use multilingual model that supports German and English
            self.coqui_tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False
            )
            
            # Check if GPU is available
            if torch.cuda.is_available():
                self.coqui_tts = self.coqui_tts.to("cuda")
                logger.info("🚀 Using GPU for TTS")
            else:
                logger.info("💻 Using CPU for TTS")
            
            self.model_name = "coqui-xtts-v2"
            logger.info("✅ Coqui TTS loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Coqui TTS: {e}")
            raise
    
    def _init_fallback_tts(self):
        """Initialize fallback TTS (pyttsx3)"""
        try:
            if not PYTTSX3_AVAILABLE:
                raise ImportError("pyttsx3 not available")
            
            logger.info("🔄 Loading pyttsx3 TTS...")
            
            self.tts_engine = pyttsx3.init()
            
            # Configure voice settings
            voices = self.tts_engine.getProperty('voices')
            
            # Try to find German voice
            german_voice = None
            for voice in voices:
                if 'german' in voice.name.lower() or 'de' in voice.id.lower():
                    german_voice = voice
                    break
            
            if german_voice:
                self.tts_engine.setProperty('voice', german_voice.id)
                logger.info(f"🇩🇪 Using German voice: {german_voice.name}")
            else:
                logger.info("🔊 Using default voice")
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 180)  # Speed
            self.tts_engine.setProperty('volume', 0.9)  # Volume
            
            self.model_name = "pyttsx3-fallback"
            logger.info("✅ pyttsx3 TTS loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load pyttsx3 TTS: {e}")
            raise
    
    async def synthesize(self, text: str, language: Optional[str] = None) -> bytes:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            language: Language code (auto-detected if None)
            
        Returns:
            Audio data as bytes
        """
        try:
            # Detect language if not provided
            if not language:
                language = self._detect_language(text)
            
            logger.info(f"🔊 Synthesizing speech (language: {language})")
            
            # Run synthesis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None, 
                self._synthesize_sync, 
                text, 
                language
            )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ TTS synthesis error: {e}")
            raise
    
    def _synthesize_sync(self, text: str, language: str) -> bytes:
        """Synchronous speech synthesis"""
        try:
            if self.coqui_tts:
                return self._synthesize_coqui(text, language)
            elif self.tts_engine:
                return self._synthesize_pyttsx3(text)
            else:
                raise RuntimeError("No TTS engine available")
                
        except Exception as e:
            logger.error(f"❌ Sync synthesis error: {e}")
            raise
    
    def _synthesize_coqui(self, text: str, language: str) -> bytes:
        """Synthesize speech using Coqui TTS"""
        try:
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Synthesize speech
            self.coqui_tts.tts_to_file(
                text=text,
                file_path=temp_path,
                language=language
            )
            
            # Read audio data
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Coqui synthesis error: {e}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    def _synthesize_pyttsx3(self, text: str) -> bytes:
        """Synthesize speech using pyttsx3"""
        try:
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save to file using pyttsx3
            self.tts_engine.save_to_file(text, temp_path)
            self.tts_engine.runAndWait()
            
            # Read audio data
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ pyttsx3 synthesis error: {e}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of input text
        
        Args:
            text: Input text
            
        Returns:
            Language code (de, en, etc.)
        """
        try:
            detected = langdetect.detect(text)
            
            # Map to supported languages
            language_map = {
                'de': 'de',  # German
                'en': 'en',  # English
                'fr': 'fr',  # French
                'es': 'es',  # Spanish
                'it': 'it',  # Italian
                'nl': 'nl',  # Dutch
                'pl': 'pl',  # Polish
                'pt': 'pt',  # Portuguese
                'ru': 'ru'   # Russian
            }
            
            return language_map.get(detected, 'en')  # Default to English
            
        except Exception as e:
            logger.warning(f"⚠️ Language detection failed: {e}, defaulting to German")
            return 'de'  # Default to German for Karlsruhe context
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        if self.coqui_tts:
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
        else:
            return {
                "de": "German",
                "en": "English"
            }
    
    def get_model_info(self):
        """Get model information"""
        return {
            "model_name": self.model_name,
            "engine": "coqui" if self.coqui_tts else "pyttsx3",
            "supported_languages": list(self.get_supported_languages().keys()),
            "gpu_available": torch.cuda.is_available() if COQUI_AVAILABLE else False,
            "status": "loaded"
        }
    
    def set_voice_settings(self, rate: int = 180, volume: float = 0.9):
        """
        Set voice settings for pyttsx3 engine
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        if self.tts_engine:
            try:
                self.tts_engine.setProperty('rate', rate)
                self.tts_engine.setProperty('volume', volume)
                logger.info(f"🔧 Voice settings updated: rate={rate}, volume={volume}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to update voice settings: {e}")
    
    def list_available_voices(self):
        """List all available voices (for pyttsx3)"""
        if self.tts_engine:
            try:
                voices = self.tts_engine.getProperty('voices')
                voice_list = []
                for voice in voices:
                    voice_info = {
                        "id": voice.id,
                        "name": voice.name,
                        "languages": getattr(voice, 'languages', []),
                        "gender": getattr(voice, 'gender', 'unknown')
                    }
                    voice_list.append(voice_info)
                return voice_list
            except Exception as e:
                logger.error(f"❌ Failed to list voices: {e}")
                return []
        else:
            return []
    
    def set_voice_by_id(self, voice_id: str):
        """
        Set voice by ID (for pyttsx3)
        
        Args:
            voice_id: Voice ID to use
        """
        if self.tts_engine:
            try:
                self.tts_engine.setProperty('voice', voice_id)
                logger.info(f"🎤 Voice changed to: {voice_id}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to set voice {voice_id}: {e}")
                return False
        return False
    
    def test_synthesis(self, test_text: str = "Dies ist ein Test der deutschen Sprachausgabe."):
        """
        Test the TTS synthesis with a sample text
        
        Args:
            test_text: Text to synthesize for testing
            
        Returns:
            Success status and audio data length
        """
        try:
            # Run sync version for testing
            audio_data = self._synthesize_sync(test_text, "de")
            logger.info(f"✅ TTS test successful: {len(audio_data)} bytes generated")
            return True, len(audio_data)
            
        except Exception as e:
            logger.error(f"❌ TTS test failed: {e}")
            return False, 0
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.tts_engine:
                # pyttsx3 cleanup
                try:
                    self.tts_engine.stop()
                except:
                    pass
                self.tts_engine = None
            
            if self.coqui_tts:
                # Coqui TTS cleanup
                try:
                    if hasattr(self.coqui_tts, 'synthesizer'):
                        del self.coqui_tts.synthesizer
                except:
                    pass
                self.coqui_tts = None
            
            logger.info("🧹 TTS service cleaned up")
            
        except Exception as e:
            logger.warning(f"⚠️ TTS cleanup warning: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass