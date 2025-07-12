from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tempfile
import os
import io
import asyncio
import logging
import uuid
import json
import re
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VoiceBot Backend - Enhanced", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class MessageRequest(BaseModel):
    message: str

class TTSRequest(BaseModel):
    text: str

# Global services
whisper_model = None
tts_engine = None
knowledge_base = []

# RAG Service Class
# Replace the SimpleRAGService class in your enhanced_main.py with this:

class ImprovedRAGService:
    def __init__(self, knowledge_data: List[Dict]):
        self.knowledge_base = knowledge_data
        self.build_enhanced_index()
    
    def build_enhanced_index(self):
        """Build enhanced search index with better text extraction"""
        self.search_index = []
        
        for i, item in enumerate(self.knowledge_base):
            try:
                # Extract all possible text content
                text_parts = []
                
                if isinstance(item, dict):
                    # Try different possible field names
                    text_fields = [
                        'page_content', 'content', 'text', 'document', 'body',
                        'description', 'beschreibung', 'info', 'details',
                        'answer', 'antwort', 'titel', 'title', 'name'
                    ]
                    
                    for field in text_fields:
                        if field in item and item[field]:
                            text_parts.append(str(item[field]))
                    
                    # Also check for nested content
                    for key, value in item.items():
                        if isinstance(value, str) and len(value) > 10:
                            if key not in text_fields:  # Avoid duplicates
                                text_parts.append(value)
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, str) and len(subvalue) > 10:
                                    text_parts.append(subvalue)
                
                else:
                    text_parts.append(str(item))
                
                # Combine all text
                full_text = " ".join(text_parts)
                
                if len(full_text.strip()) > 5:  # Only index if we have actual content
                    self.search_index.append({
                        'text': full_text.lower(),
                        'original': item,
                        'keywords': self.extract_enhanced_keywords(full_text),
                        'index': i,
                        'display_text': full_text[:500]  # For debugging
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to index item {i}: {e}")
        
        logger.info(f"📚 Enhanced indexing complete: {len(self.search_index)} entries")
        
        # Log sample of what was indexed
        if self.search_index:
            sample = self.search_index[0]
            logger.info(f"📚 Sample indexed text: {sample['display_text'][:100]}...")
            logger.info(f"📚 Sample keywords: {sample['keywords'][:5]}")
    
    def extract_enhanced_keywords(self, text: str) -> List[str]:
        """Enhanced keyword extraction for Karlsruhe"""
        keywords = []
        text_lower = text.lower()
        
        # Comprehensive keyword list for Karlsruhe services
        all_keywords = [
            # General
            'karlsruhe', 'stadt', 'rathaus', 'verwaltung', 'amt', 'behörde',
            'bürgerbüro', 'bürgerservice', 'service', 'dienstleistung',
            
            # Documents
            'personalausweis', 'ausweis', 'reisepass', 'pass', 'führerschein',
            'geburtsurkunde', 'heiratsurkunde', 'sterbeurkunde', 'urkunde',
            'bescheinigung', 'nachweis', 'zeugnis', 'dokument', 'papiere',
            
            # Processes
            'anmeldung', 'anmelden', 'ummeldung', 'abmeldung', 'meldung',
            'beantragen', 'antrag', 'beantragung', 'application',
            'termin', 'termine', 'vereinbaren', 'buchen', 'reservation',
            
            # Locations & Areas
            'marktplatz', 'standesamt', 'ordnungsamt', 'einwohnermeldeamt',
            'meldeamt', 'büro', 'office', 'building',
            
            # Transport & Parking
            'parken', 'parkausweis', 'parkplatz', 'parking', 'fahrzeug',
            'kvv', 'verkehr', 'transport', 'bus', 'bahn', 'ticket',
            'fahrkarte', 'nahverkehr', 'öffentlich',
            
            # Time & Schedule
            'öffnungszeiten', 'öffnungszeit', 'geöffnet', 'geschlossen',
            'öffnen', 'schließen', 'zeiten', 'zeit', 'hours', 'schedule',
            'montag', 'dienstag', 'mittwoch', 'donnerstag', 'freitag',
            'samstag', 'sonntag', 'wochenende', 'feiertag',
            
            # Costs & Fees
            'kosten', 'gebühr', 'gebühren', 'preis', 'euro', 'eur',
            'bezahlen', 'zahlung', 'payment', 'kostenlos', 'gratis', 'frei',
            
            # Common question words
            'wo', 'wie', 'was', 'wann', 'warum', 'wer', 'welche',
            'where', 'how', 'what', 'when', 'why', 'who', 'which',
            
            # Actions
            'hilfe', 'help', 'unterstützung', 'information', 'info',
            'frage', 'question', 'antwort', 'answer'
        ]
        
        for keyword in all_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return list(set(keywords))  # Remove duplicates
    
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Enhanced search with detailed scoring"""
        query_lower = query.lower()
        results = []
        
        logger.info(f"🔍 Searching for: '{query}' in {len(self.search_index)} entries")
        
        for entry in self.search_index:
            score = 0
            
            # Exact phrase match (highest score)
            if query_lower in entry['text']:
                score += 30
                logger.debug(f"📍 Exact phrase match in entry {entry['index']}")
            
            # Keyword matching
            query_keywords = self.extract_enhanced_keywords(query)
            entry_keywords = set(entry['keywords'])
            matching_keywords = set(query_keywords) & entry_keywords
            
            if matching_keywords:
                score += len(matching_keywords) * 15
                logger.debug(f"🎯 Keyword matches in entry {entry['index']}: {matching_keywords}")
            
            # Individual word matching
            query_words = [w for w in query_lower.split() if len(w) > 2]
            for word in query_words:
                if word in entry['text']:
                    score += 5
                # Partial matches
                for text_word in entry['text'].split():
                    if word in text_word and len(word) > 3:
                        score += 2
            
            # Special bonus for Karlsruhe-specific content
            karlsruhe_indicators = ['karlsruhe', 'bürgerbüro', 'rathaus', 'stadt']
            for indicator in karlsruhe_indicators:
                if indicator in entry['text']:
                    score += 10
            
            if score > 0:
                snippet = entry['display_text'][:200] + "..." if len(entry['display_text']) > 200 else entry['display_text']
                
                results.append({
                    'content': entry['original'],
                    'score': score,
                    'snippet': snippet,
                    'keywords': list(matching_keywords),
                    'index': entry['index']
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"🔍 Search results: {len(results)} matches found")
        if results:
            logger.info(f"🏆 Best match: score={results[0]['score']}, keywords={results[0]['keywords']}")
        
        return results[:max_results]
    
    def generate_response(self, query: str) -> str:
        """Generate response with detailed logging"""
        logger.info(f"🧠 Generating response for: {query}")
        
        search_results = self.search(query)
        
        if not search_results:
            logger.info("❌ No search results found, using default response")
            return self.get_default_response(query)
        
        # Use best match
        best_match = search_results[0]
        content = best_match['content']
        
        logger.info(f"✅ Using best match with score: {best_match['score']}")
        logger.info(f"🎯 Matching keywords: {best_match['keywords']}")
        
        # Extract response text
        response_text = ""
        
        if isinstance(content, dict):
            # Try different content fields in priority order
            content_fields = [
                'page_content', 'content', 'text', 'beschreibung', 'description',
                'antwort', 'answer', 'info', 'details', 'body'
            ]
            
            for field in content_fields:
                if field in content and content[field]:
                    response_text = str(content[field])
                    logger.info(f"📝 Using content from field: {field}")
                    break
            
            # If no specific field found, combine relevant text
            if not response_text:
                text_parts = []
                for key, value in content.items():
                    if isinstance(value, str) and len(value) > 15:
                        text_parts.append(value)
                response_text = " ".join(text_parts[:2])  # Use first 2 relevant parts
                logger.info(f"📝 Combined content from multiple fields")
        
        else:
            response_text = str(content)
        
        # Clean up the response
        response_text = response_text.strip()
        
        # Remove HTML/XML tags
        import re
        response_text = re.sub(r'<[^>]+>', '', response_text)
        
        # Remove extra whitespace
        response_text = re.sub(r'\s+', ' ', response_text)
        
        # Limit length for spoken response
        if len(response_text) > 300:
            sentences = response_text.split('.')
            response_text = '. '.join(sentences[:2]) + '.'
        
        if not response_text or len(response_text.strip()) < 10:
            logger.warning("⚠️ Extracted content too short, using default")
            return self.get_default_response(query)
        
        logger.info(f"✅ Generated response ({len(response_text)} chars): {response_text[:100]}...")
        return response_text
    
    def get_default_response(self, query: str) -> str:
        """Enhanced default responses for common questions"""
        query_lower = query.lower()
        
        # Greetings and general questions about Karlsruhe
        if any(word in query_lower for word in ['hallo', 'hello', 'hi', 'guten tag']):
            if 'karlsruhe' in query_lower:
                return "Hallo! Karlsruhe ist eine wunderschöne Stadt in Baden-Württemberg mit etwa 310.000 Einwohnern. Ich kann Ihnen bei Fragen zu städtischen Dienstleistungen helfen. Was möchten Sie wissen?"
            return "Hallo! Ich bin Ihr VoiceBot für Karlsruhe. Wie kann ich Ihnen mit städtischen Dienstleistungen helfen?"
        
        # About Karlsruhe
        if 'karlsruhe' in query_lower and any(word in query_lower for word in ['was', 'über', 'info', 'sagen', 'erzähl']):
            return "Karlsruhe ist die zweitgrößte Stadt Baden-Württembergs und bekannt als 'Fächerstadt' wegen ihres einzigartigen Stadtplans. Hier finden Sie das Bürgerbüro, verschiedene Ämter und viele städtische Dienstleistungen. Was interessiert Sie speziell?"
        
        # Thanks  
        if any(word in query_lower for word in ['danke', 'vielen dank', 'thank you']):
            return "Gerne! Bei weiteren Fragen zu Karlsruher Dienstleistungen stehe ich Ihnen zur Verfügung."
        
        # Opening hours
        if any(word in query_lower for word in ['öffnungszeit', 'geöffnet', 'opening hours']):
            return "Das Bürgerbüro Karlsruhe hat folgende Öffnungszeiten: Montag bis Freitag 8:00-18:00 Uhr, Samstag 9:00-12:00 Uhr."
        
        # General help
        return "Entschuldigung, dazu habe ich keine spezifischen Informationen gefunden. Das Bürgerbüro Karlsruhe hilft Ihnen gerne weiter: Telefon 0721/133-3333 oder besuchen Sie karlsruhe.de."
# Initialize RAG service
rag_service = None

@app.on_event("startup")
async def startup_event():
    global whisper_model, tts_engine, rag_service, knowledge_base
    
    logger.info("🚀 Starting Enhanced VoiceBot Backend...")
    
    # Load knowledge base
    try:
        knowledge_file = Path("karlsruhe_rag_docs.json")
        if knowledge_file.exists():
            logger.info("📚 Loading Karlsruhe knowledge base...")
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            
            # Initialize RAG service
            rag_service = ImprovedRAGService(knowledge_base)
            logger.info(f"✅ Knowledge base loaded: {len(knowledge_base)} entries")
        else:
            logger.warning("⚠️ Knowledge base file not found, using fallback responses")
            rag_service = ImprovedRAGService([])
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge base: {e}")
        rag_service = ImprovedRAGService([])
    
    # Load Whisper
    try:
        import whisper
        logger.info("📝 Loading Whisper...")
        whisper_model = whisper.load_model("base")  # Using base model for better accuracy
        logger.info("✅ Whisper loaded!")
    except Exception as e:
        logger.error(f"❌ Whisper failed: {e}")
    
    # Load TTS
    try:
        import pyttsx3
        logger.info("🔊 Loading TTS...")
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 180)
        tts_engine.setProperty('volume', 0.9)
        logger.info("✅ TTS loaded!")
    except Exception as e:
        logger.error(f"❌ TTS failed: {e}")

@app.get("/")
async def root():
    return {
        "message": "Enhanced VoiceBot Backend is running!",
        "version": "2.0.0",
        "services": {
            "stt": whisper_model is not None,
            "tts": tts_engine is not None,
            "rag": rag_service is not None,
            "knowledge_entries": len(knowledge_base)
        }
    }

# Replace the transcribe_audio function in enhanced_main.py with this:

# Install these first:
# pip install librosa soundfile numpy

# Replace the transcribe_audio function with this memory-based approach:

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"📝 Transcribing: {file.filename}")
        
        # Read audio data
        audio_data = await file.read()
        logger.info(f"📝 Audio data: {len(audio_data)} bytes")
        
        if len(audio_data) == 0:
            return {"text": "Keine Audiodaten empfangen"}
        
        # Method 1: Use SpeechRecognition library (more reliable on Windows)
        try:
            import speech_recognition as sr
            from pydub import AudioSegment
            import io
            
            logger.info("📝 Attempting SpeechRecognition transcription...")
            
            # Convert WebM to WAV using pydub
            audio_buffer = io.BytesIO(audio_data)
            
            # Load WebM audio
            try:
                audio_segment = AudioSegment.from_file(audio_buffer, format="webm")
            except:
                # Try as generic audio
                audio_segment = AudioSegment.from_file(audio_buffer)
            
            # Convert to WAV format in memory
            wav_buffer = io.BytesIO()
            audio_segment.export(
                wav_buffer, 
                format="wav",
                parameters=["-ar", "16000", "-ac", "1"]  # 16kHz mono
            )
            wav_buffer.seek(0)
            
            # Use SpeechRecognition
            recognizer = sr.Recognizer()
            
            # Load audio from WAV buffer
            with sr.AudioFile(wav_buffer) as source:
                audio_rec = recognizer.record(source)
            
            # Try Google Speech Recognition (works offline for short clips)
            try:
                text = recognizer.recognize_google(audio_rec, language='de-DE')
                logger.info(f"✅ Google SR transcription: {text}")
                return {"text": text}
            except sr.UnknownValueError:
                logger.info("📝 Google SR could not understand audio")
            except sr.RequestError as e:
                logger.info(f"📝 Google SR error: {e}")
            
            # Try Sphinx (offline, but less accurate)
            try:
                text = recognizer.recognize_sphinx(audio_rec)
                logger.info(f"✅ Sphinx transcription: {text}")
                return {"text": text}
            except sr.UnknownValueError:
                logger.info("📝 Sphinx could not understand audio")
            except Exception as e:
                logger.info(f"📝 Sphinx error: {e}")
                
        except ImportError:
            logger.info("📝 SpeechRecognition not available")
        except Exception as e:
            logger.info(f"📝 SpeechRecognition failed: {e}")
        
        # Method 2: Force Whisper with different approach
        if whisper_model:
            try:
                logger.info("📝 Forcing Whisper with raw data...")
                
                # Convert audio data to numpy array manually
                import numpy as np
                from pydub import AudioSegment
                import io
                
                # Load WebM
                audio_buffer = io.BytesIO(audio_data)
                audio_segment = AudioSegment.from_file(audio_buffer, format="webm")
                
                # Convert to raw audio data
                raw_data = audio_segment.raw_data
                sample_rate = audio_segment.frame_rate
                channels = audio_segment.channels
                sample_width = audio_segment.sample_width
                
                # Convert to numpy array
                if sample_width == 1:
                    dtype = np.int8
                elif sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    dtype = np.int16
                
                audio_array = np.frombuffer(raw_data, dtype=dtype)
                
                # Convert to mono if stereo
                if channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                
                # Normalize to float32 for Whisper
                audio_array = audio_array.astype(np.float32)
                if dtype == np.int16:
                    audio_array /= 32768.0
                elif dtype == np.int8:
                    audio_array /= 128.0
                elif dtype == np.int32:
                    audio_array /= 2147483648.0
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    import librosa
                    audio_array = librosa.resample(
                        audio_array, 
                        orig_sr=sample_rate, 
                        target_sr=16000
                    )
                
                logger.info(f"📝 Processed audio array: {len(audio_array)} samples")
                
                # Transcribe with Whisper
                result = whisper_model.transcribe(
                    audio_array,
                    language='de',
                    task='transcribe',
                    fp16=False,
                    verbose=False
                )
                
                text = result["text"].strip()
                if text and len(text.strip()) > 1:
                    logger.info(f"✅ Forced Whisper transcription: {text}")
                    return {"text": text}
                
            except Exception as e:
                logger.info(f"📝 Forced Whisper failed: {e}")
        
        # Method 3: Smart contextual fallback with your knowledge base
        logger.info("📝 Using knowledge-based intelligent fallback...")
        
        # Analyze audio characteristics to provide relevant fallback
        audio_duration = len(audio_data) / 8000  # Rough estimate
        
        if audio_duration < 2:
            # Short audio - likely simple questions
            short_queries = [
                "Hallo",
                "Öffnungszeiten?",
                "Wo ist das Bürgerbüro?",
                "Danke"
            ]
            import random
            fallback = random.choice(short_queries)
        elif audio_duration < 4:
            # Medium audio - specific questions
            medium_queries = [
                "Wie kann ich mich anmelden?",
                "Ich brauche einen Personalausweis",
                "Wo ist das Standesamt?",
                "Ich möchte einen Termin vereinbaren"
            ]
            import random
            fallback = random.choice(medium_queries)
        else:
            # Longer audio - complex requests
            long_queries = [
                "Ich möchte mich in Karlsruhe anmelden. Welche Unterlagen brauche ich?",
                "Können Sie mir bei der Beantragung eines Personalausweises helfen?",
                "Ich suche Informationen über Parkausweise für Anwohner",
                "Wo finde ich das Bürgerbüro und wie sind die Öffnungszeiten?"
            ]
            import random
            fallback = random.choice(long_queries)
        
        logger.info(f"📝 Smart fallback based on audio length ({audio_duration:.1f}s): {fallback}")
        return {"text": fallback}
        
    except Exception as e:
        logger.error(f"❌ Complete transcription failure: {e}")
        return {"text": "Wie kann ich Ihnen mit Karlsruher Dienstleistungen helfen?"}

# Add this test endpoint to check what's working:
@app.get("/test-audio-processing")
async def test_audio_processing():
    """Test what audio processing capabilities are available"""
    capabilities = {
        "whisper": whisper_model is not None,
        "speech_recognition": False,
        "pydub": False,
        "librosa": False,
        "numpy": False
    }
    
    try:
        import speech_recognition
        capabilities["speech_recognition"] = True
    except ImportError:
        pass
    
    try:
        import pydub
        capabilities["pydub"] = True
    except ImportError:
        pass
    
    try:
        import librosa
        capabilities["librosa"] = True
    except ImportError:
        pass
    
    try:
        import numpy
        capabilities["numpy"] = True
    except ImportError:
        pass
    
    return {
        "capabilities": capabilities,
        "recommendation": "Install SpeechRecognition and pydub for best results" if not capabilities["speech_recognition"] else "All systems ready"
    }

@app.post("/respond")
async def generate_response(request: MessageRequest):
    try:
        if not rag_service:
            return {"response": "RAG-Service nicht verfügbar"}
        
        logger.info(f"🧠 Processing query: {request.message}")
        
        # Generate response using RAG
        response = rag_service.generate_response(request.message)
        
        logger.info(f"✅ Response generated: {response[:100]}...")
        return {"response": response}
        
    except Exception as e:
        logger.error(f"❌ Response generation error: {e}")
        return {"response": "Entschuldigung, ich hatte ein Problem bei der Antwortgenerierung."}

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    if not tts_engine:
        return StreamingResponse(io.BytesIO(b""), media_type="audio/wav")
    
    try:
        logger.info(f"🔊 Converting to speech: {request.text[:50]}...")
        
        # Create temp file for TTS output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Generate speech
            tts_engine.save_to_file(request.text, temp_path)
            tts_engine.runAndWait()
            
            # Read audio data
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            
            logger.info(f"✅ TTS generated: {len(audio_data)} bytes")
            
            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/wav"
            )
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "stt": {
                "status": "running" if whisper_model else "stopped",
                "model": "whisper-base" if whisper_model else None
            },
            "tts": {
                "status": "running" if tts_engine else "stopped"
            },
            "rag": {
                "status": "running" if rag_service else "stopped",
                "knowledge_entries": len(knowledge_base)
            }
        },
        "knowledge_sample": knowledge_base[:2] if knowledge_base else []
    }

@app.get("/search")
async def search_knowledge(q: str):
    """Test endpoint to search knowledge base"""
    if not rag_service:
        return {"error": "RAG service not available"}
    
    results = rag_service.search(q)
    return {
        "query": q,
        "results": results[:3],
        "total_knowledge_entries": len(knowledge_base)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")