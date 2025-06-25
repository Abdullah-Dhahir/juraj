    
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from voice_transcriber import transcribe_from_microphone

app = FastAPI()

# Enable CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "VoiceBot API is running!"}

@app.post("/transcribe")
def transcribe():
    """
    Capture voice from microphone, transcribe it, and return the result.
    """
    transcript = transcribe_from_microphone()
    return {"transcript": transcript}

# Placeholder for Ivan's LLM response module
@app.post("/respond")
def respond(payload: dict):
    user_input = payload.get("text", "")
    # Replace this with a call to Start_Bot or LangChain pipeline
    return {"response": f"(LLM Response to: {user_input})"}

# Placeholder for Fatima's TTS module
@app.post("/speak")
def speak(payload: dict):
    response_text = payload.get("text", "")
    # Replace this with actual TTS code from TTS_2.py
    return {"audio_path": f"/static/audio/{response_text[:10]}.wav"}
