from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import logging
import io
from pathlib import Path

# Import our service modules
from stt_service import STTService
from nlp_service import NLPService
from tts_service import TTSService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="VoiceBot Backend", version="1.0.0")

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests
class MessageRequest(BaseModel):
    message: str

class TTSRequest(BaseModel):
    text: str

# Initialize services
stt_service = None
nlp_service = None
tts_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    global stt_service, nlp_service, tts_service
    
    logger.info("🚀 Starting VoiceBot Backend...")
    
    try:
        # Initialize STT Service
        logger.info("📝 Loading STT Service (Whisper)...")
        stt_service = STTService()
        
        # Initialize NLP Service
        logger.info("🧠 Loading NLP Service (RAG + LLM)...")
        nlp_service = NLPService()
        
        # Initialize TTS Service
        logger.info("🔊 Loading TTS Service (Coqui)...")
        tts_service = TTSService()
        
        logger.info("✅ All services loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "VoiceBot Backend is running!",
        "version": "1.0.0",
        "services": {
            "stt": stt_service is not None,
            "nlp": nlp_service is not None,
            "tts": tts_service is not None
        }
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file to text using STT service
    """
    try:
        logger.info(f"📝 Transcribing audio file: {file.filename}")
        
        if not stt_service:
            raise HTTPException(status_code=503, detail="STT service not available")
        
        # Read audio file
        audio_data = await file.read()
        
        # Transcribe audio
        transcription = await stt_service.transcribe(audio_data)
        
        logger.info(f"✅ Transcription completed: {transcription[:50]}...")
        
        return {"text": transcription}
        
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/respond")
async def generate_response(request: MessageRequest):
    """
    Generate bot response using NLP service
    """
    try:
        logger.info(f"🧠 Processing message: {request.message[:50]}...")
        
        if not nlp_service:
            raise HTTPException(status_code=503, detail="NLP service not available")
        
        # Generate response
        response = await nlp_service.generate_response(request.message)
        
        logger.info(f"✅ Response generated: {response[:50]}...")
        
        return {"response": response}
        
    except Exception as e:
        logger.error(f"❌ Response generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using TTS service
    """
    try:
        logger.info(f"🔊 Converting text to speech: {request.text[:50]}...")
        
        if not tts_service:
            raise HTTPException(status_code=503, detail="TTS service not available")
        
        # Generate speech
        audio_data = await tts_service.synthesize(request.text)
        
        logger.info("✅ Speech synthesis completed")
        
        # Return audio as streaming response
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check for all services"""
    return {
        "status": "healthy",
        "services": {
            "stt": {
                "status": "running" if stt_service else "stopped",
                "model": stt_service.model_name if stt_service else None
            },
            "nlp": {
                "status": "running" if nlp_service else "stopped",
                "model": nlp_service.model_name if nlp_service else None
            },
            "tts": {
                "status": "running" if tts_service else "stopped",
                "model": tts_service.model_name if tts_service else None
            }
        }
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )