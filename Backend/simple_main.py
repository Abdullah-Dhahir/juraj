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

# Add after other imports
try:
    from pydub import AudioSegment
    AUDIO_CONVERSION_AVAILABLE = True
except ImportError:
    AUDIO_CONVERSION_AVAILABLE = False
    print("⚠️ pydub not available - audio conversion disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VoiceBot Backend", version="1.0.0")

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

# Initialize services
whisper_model = None
tts_engine = None

@app.on_event("startup")
async def startup_event():
    global whisper_model, tts_engine
    
    logger.info("🚀 Starting VoiceBot Backend...")
    
    # Load Whisper
    try:
        import whisper
        logger.info("📝 Loading Whisper...")
        whisper_model = whisper.load_model("base")
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
        "message": "VoiceBot Backend is running!",
        "version": "1.0.0",
        "services": {
            "stt": whisper_model is not None,
            "tts": tts_engine is not None,
            "nlp": True  # Simple responses
        }
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not whisper_model:
        return {"text": "Hallo, das ist ein Test"}
    
    temp_path = None
    wav_path = None
    try:
        logger.info(f"📝 Transcribing audio file: {file.filename}")
        
        # Read audio file
        audio_data = await file.read()
        logger.info(f"📝 Audio data received: {len(audio_data)} bytes")
        
        if len(audio_data) == 0:
            return {"text": "Keine Audiodaten empfangen"}
        
        # Create temporary file for original audio
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_data)
            tmp.flush()
            temp_path = tmp.name
        
        logger.info(f"📝 Original file created: {temp_path}")
        
        # Convert to WAV if pydub is available
        if AUDIO_CONVERSION_AVAILABLE:
            try:
                # Load audio with pydub (supports many formats)
                audio = AudioSegment.from_file(temp_path)
                
                # Convert to WAV
                wav_path = temp_path.replace('.webm', '.wav')
                audio.export(wav_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                
                logger.info(f"📝 Converted to WAV: {wav_path}")
                logger.info(f"📝 WAV file exists: {os.path.exists(wav_path)}")
                logger.info(f"📝 WAV file size: {os.path.getsize(wav_path)} bytes")
                
                # Use the converted WAV file
                transcribe_file = wav_path
                
            except Exception as conv_error:
                logger.error(f"❌ Audio conversion failed: {conv_error}")
                # Fall back to original file
                transcribe_file = temp_path
        else:
            # No conversion available, use original file
            transcribe_file = temp_path
        
        # Try transcription
        logger.info(f"📝 Attempting transcription of: {transcribe_file}")
        result = whisper_model.transcribe(transcribe_file)
        text = result["text"].strip()
        
        # Cleanup files
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            logger.info(f"📝 Cleanup completed")
        except:
            pass
        
        logger.info(f"✅ Transcription successful: {text}")
        return {"text": text}
        
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        
        # Cleanup on error
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except:
            pass
        
        return {"text": "Entschuldigung, Transkription fehlgeschlagen"}
    
@app.post("/respond")
async def generate_response(request: MessageRequest):
    try:
        message = request.message.lower()
        
        # Simple rule-based responses for Karlsruhe
        if "hallo" in message or "hello" in message:
            response = "Hallo! Ich bin Ihr VoiceBot für Karlsruhe. Wie kann ich Ihnen helfen?"
        elif "bürgerbüro" in message or "citizen office" in message:
            response = "Das Bürgerbüro Karlsruhe befindet sich im Rathaus am Marktplatz. Öffnungszeiten: Mo-Fr 8:00-18:00, Sa 9:00-12:00."
        elif "anmeld" in message or "register" in message:
            response = "Für die Anmeldung in Karlsruhe benötigen Sie: Personalausweis oder Reisepass, Wohnungsgeberbestätigung. Die Anmeldung muss innerhalb von 14 Tagen erfolgen."
        elif "öffnungszeit" in message or "opening hours" in message:
            response = "Das Bürgerbüro ist geöffnet: Montag bis Freitag 8:00-18:00 Uhr, Samstag 9:00-12:00 Uhr."
        elif "parken" in message or "parking" in message:
            response = "Parkausweise für Anwohner können im Ordnungsamt beantragt werden. Kosten: 30€ pro Jahr. Benötigt: Fahrzeugschein, Personalausweis, Meldebescheinigung."
        elif "standesamt" in message or "civil registry" in message:
            response = "Das Standesamt Karlsruhe führt Eheschließungen und Beurkundungen durch. Terminvereinbarung erforderlich unter Tel. 0721/133-5301."
        else:
            response = "Entschuldigung, zu diesem Thema habe ich keine spezifischen Informationen. Können Sie Ihre Frage zu Karlsruher Dienstleistungen präzisieren?"
        
        logger.info(f"✅ Response generated for: {message[:30]}...")
        return {"response": response}
        
    except Exception as e:
        logger.error(f"❌ Response error: {e}")
        return {"response": "Entschuldigung, ich hatte ein Problem bei der Antwort."}

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    if not tts_engine:
        # Return empty audio
        return StreamingResponse(
            io.BytesIO(b""),
            media_type="audio/wav"
        )
    
    try:
        logger.info(f"🔊 Converting to speech: {request.text[:50]}...")
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Generate speech
        tts_engine.save_to_file(request.text, temp_path)
        tts_engine.runAndWait()
        
        # Read audio data
        with open(temp_path, "rb") as f:
            audio_data = f.read()
        
        # Cleanup
        os.unlink(temp_path)
        
        logger.info("✅ Speech generated!")
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav"
        )
        
    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "stt": {"status": "running" if whisper_model else "stopped"},
            "tts": {"status": "running" if tts_engine else "stopped"},
            "nlp": {"status": "running"}
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")