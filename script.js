// =============================
// VoiceBot – Final Combined Script
// =============================

let mediaRecorder;
let audioChunks = [];

// === Aufnahme starten ===
document.getElementById("recordBtn").addEventListener("click", async () => {
  document.getElementById("recordBtn").disabled = true;
  document.getElementById("userInput").textContent = "🎙️ Aufnahme läuft...";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = event => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      const formData = new FormData();
      formData.append("file", audioBlob, "voice.wav");

      document.getElementById("userInput").textContent = "⏳ Transkription läuft...";

      const response = await fetch("http://localhost:8000/transcribe", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      const text = data.text;

      document.getElementById("userInput").textContent = text;

      // Antwort vom Bot holen
      const res2 = await fetch("http://localhost:8000/respond", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      });

      const result = await res2.json();
      document.getElementById("botResponse").textContent = result.response;

      document.getElementById("recordBtn").disabled = false;
      document.getElementById("recordBtn").textContent = "🎤 Start Aufnahme";
    };

    mediaRecorder.start();

    setTimeout(() => {
      mediaRecorder.stop();
      document.getElementById("recordBtn").textContent = "⏹️ Aufnahme beendet";
    }, 5000);

  } catch (error) {
    console.error("❌ Mikrofonfehler:", error);
    alert("Zugriff auf Mikrofon fehlgeschlagen.");
    document.getElementById("recordBtn").disabled = false;
  }
});

// === TTS abspielen ===
document.getElementById("playBtn").addEventListener("click", async () => {
  const text = document.getElementById("botResponse").textContent;

  if (!text || text.trim() === "") {
    console.warn("⚠️ No text to speak.");
    document.getElementById("botResponse").textContent = "⚠️ Kein Text zum Vorlesen.";
    return;
  }

  document.getElementById("playBtn").textContent = "🔄 Lade TTS...";
  document.getElementById("playBtn").disabled = true;

  try {
    const response = await fetch("http://localhost:8000/tts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: text })
    });

    if (!response.ok) {
      throw new Error("TTS-Server antwortet nicht");
    }

    const blob = await response.blob();
    const audioURL = URL.createObjectURL(blob);
    const audio = new Audio(audioURL);
    audio.play();

  } catch (err) {
    console.error("❌ TTS playback error:", err);
    alert("Fehler beim Abrufen der Sprachausgabe. Ist der TTS-Server aktiv?");
  } finally {
    document.getElementById("playBtn").textContent = "🔊 Antwort anhören";
    document.getElementById("playBtn").disabled = false;
  }
});

// === Settings Button ===
document.getElementById("settingsBtn").addEventListener("click", () => {
  alert("Einstellungen sind noch nicht implementiert.");
});
