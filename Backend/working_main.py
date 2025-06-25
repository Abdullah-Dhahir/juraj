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
        whisper_model = whisper.load_model("tiny")  # Use tiny model for faster loading
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
            "nlp": True
        }
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # For now, return a mock response to test the rest of the pipeline
    logger.info(f"📝 Mock transcription for: {file.filename}")
    
    # Read the file to simulate processing
    audio_data = await file.read()
    logger.info(f"📝 Received {len(audio_data)} bytes of audio data")
    
    # Return mock German responses to test the system
    mock_responses = [
        "Hallo, wo ist das Bürgerbüro?",
        "Ich möchte mich anmelden",
        "Wie sind die Öffnungszeiten?",
        "Wo kann ich parken?",
        "Ich brauche einen Personalausweis"
    ]
    
    import random
    mock_text = random.choice(mock_responses)
    logger.info(f"📝 Mock transcription: {mock_text}")
    
    return {"text": mock_text}

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
        elif "personalausweis" in message or "id card" in message:
            response = "Personalausweise können im Bürgerbüro beantragt werden. Benötigt: Biometrisches Foto, alte Dokumente, 37€ Gebühr. Bearbeitungszeit: 2-3 Wochen."
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "stt": {"status": "mock"},
            "tts": {"status": "running" if tts_engine else "stopped"},
            "nlp": {"status": "running"}
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")