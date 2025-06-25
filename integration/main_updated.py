import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from integration.Start_Bot_refactored import generate_response
from TTS.api import TTS
from langdetect import detect

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development; limit in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Coqui TTS models
tts_en = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
tts_de = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False, gpu=False)

@app.get("/")
def root():
    return {"message": "VoiceBot API is running!"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp_uploaded.wav", "wb") as f:
        f.write(contents)
    print("📥 Datei empfangen:", file.filename)
    # Replace this with real STT logic later (Whisper, Vosk, etc.)
    return {"text": "Ich möchte einen neuen Reisepass beantragen."}

@app.post("/respond")
async def respond(request: Request):
    data = await request.json()
    user_input = data.get("text", "")
    reply = generate_response(user_input)
    return {"response": reply}

@app.post("/speak")
async def speak(request: Request):
    data = await request.json()
    response_text = data.get("text", "")
    lang = detect(response_text)
    output_path = "output.wav"

    if lang == "de":
        tts_de.tts_to_file(text=response_text, file_path=output_path)
    else:
        tts_en.tts_to_file(text=response_text, file_path=output_path)

    return FileResponse(output_path, media_type="audio/wav")
