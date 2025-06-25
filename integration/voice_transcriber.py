# ============================================
# 📦 Import Libraries
# ============================================
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import noisereduce as nr
import whisper
import logging
from datetime import datetime
from queue import Queue
from scipy.signal import butter, lfilter

# ============================================
# ⚙️ Configuration
# ============================================
SAMPLING_RATE = 16000
BLOCKSIZE = 512
SILENCE_BLOCKS = 50
VAD_THRESHOLD = 0.5
OUTPUT_DIR = r"D:\Voice-bot"

# ============================================
# 📝 Setup Logging
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# 🔊 High-Pass Filter
# ============================================
def high_pass_filter(audio, sr, cutoff=100):
    b, a = butter(1, cutoff / (sr / 2), btype='high', analog=False)
    return lfilter(b, a, audio)

# ============================================
# 🔥 Load Silero VAD Model
# ============================================
log.info("🔄 Loading Silero VAD model...")
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(get_speech_timestamps, _, _, _, _) = utils

# ============================================
# 🎙️ Auto Start/Stop Recording Using VAD
# ============================================
def record_until_silence():
    log.info("🎤 Listening... Speak now. Will stop after silence.")

    q = Queue()
    silence_count = 0
    speech_detected = False
    speech_buffer = []

    def callback(indata, frames, time_info, status):
        if status:
            log.warning(f"Audio input status: {status}")
        q.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLING_RATE, blocksize=BLOCKSIZE,
                        device=None, channels=1, dtype='float32', callback=callback):
        while True:
            block = q.get()
            audio_np = block[:, 0]
            audio_tensor = torch.from_numpy(audio_np)

            prob = model_vad(audio_tensor, SAMPLING_RATE).item()

            if prob > VAD_THRESHOLD:
                silence_count = 0
                speech_detected = True
                speech_buffer.append(audio_np)
            elif speech_detected:
                silence_count += 1
                if silence_count > SILENCE_BLOCKS:
                    log.info("🛑 Silence detected. Stopping recording.")
                    break

    return np.concatenate(speech_buffer)

# ============================================
# 🧹 Clean Audio, Run VAD, Transcribe
# ============================================
def process_and_transcribe(raw_audio, filename_base):
    raw_path = os.path.join(OUTPUT_DIR, f"{filename_base}.wav")
    sf.write(raw_path, raw_audio, SAMPLING_RATE)
    log.info(f"✅ Raw audio saved: {raw_path}")

    log.info("🧹 Reducing noise...")
    reduced = nr.reduce_noise(y=raw_audio, sr=SAMPLING_RATE, prop_decrease=0.6, stationary=True)

    log.info("🔊 Applying high-pass filter...")
    cleaned = high_pass_filter(reduced, SAMPLING_RATE)

    clean_path = os.path.join(OUTPUT_DIR, f"{filename_base}_clean.wav")
    sf.write(clean_path, cleaned.astype(np.float32), SAMPLING_RATE)
    log.info(f"✅ Cleaned audio saved: {clean_path}")

    log.info("🔍 Extracting speech-only segments...")
    waveform = torch.from_numpy(cleaned.astype(np.float32)).unsqueeze(0)
    speech_timestamps = get_speech_timestamps(
        waveform[0], model_vad, sampling_rate=SAMPLING_RATE,
        min_speech_duration_ms=250, speech_pad_ms=200
    )

    if not speech_timestamps:
        log.warning("⚠️ No speech detected.")
        return None

    speech_segments = torch.cat([
        waveform[:, ts['start']:ts['end']] for ts in speech_timestamps
    ], dim=1)

    speech_output_file = os.path.join(OUTPUT_DIR, f"{filename_base}_speech.wav")
    sf.write(speech_output_file, speech_segments.squeeze().numpy(), SAMPLING_RATE)
    log.info(f"✅ Speech-only audio saved: {speech_output_file}")

    # ============================================
    # 📝 Whisper Transcription
    # ============================================
    log.info("🔠 Loading Whisper model (base)...")
    model_whisper = whisper.load_model("base")

    log.info("📝 Transcribing...")
    result = model_whisper.transcribe(speech_output_file)

    log.info("📄 Transcription Result:")
    print("📝", result["text"])
    print(f"🌍 Detected language: {result['language']}")

    transcript_path = os.path.join(OUTPUT_DIR, f"{filename_base}_transcription.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    return result["text"]

# ============================================
# ✅ Callable Function
# ============================================
def transcribe_from_microphone():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"recording_{timestamp}"

    raw_audio = record_until_silence()
    if raw_audio is not None:
        return process_and_transcribe(raw_audio, filename_base)
    else:
        return "❌ No audio recorded."

# ============================================
# ▶️ Run Standalone
# ============================================
if __name__ == "__main__":
    transcription = transcribe_from_microphone()
    print("\n✅ Final Result:", transcription)
