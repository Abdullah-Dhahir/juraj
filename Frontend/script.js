// =============================
// VoiceBot - Clean Single File Frontend
// =============================

let mediaRecorder = null;
let audioChunks = [];
let currentStream = null;
let isRecording = false;

// =============================
// Initialization
// =============================

document.addEventListener('DOMContentLoaded', function() {
    console.log("🎙️ VoiceBot loading...");
    
    // Setup event listeners
    document.getElementById("recordBtn").addEventListener("click", toggleRecording);
    document.getElementById("playBtn").addEventListener("click", playResponse);
    document.getElementById("settingsBtn").addEventListener("click", showSettings);
    
    // Keyboard shortcuts
    document.addEventListener("keydown", handleKeyPress);
    
    // Check backend status
    checkBackendStatus();
    
    console.log("✅ VoiceBot ready!");
    console.log("📱 Audio support:", {
        webm: MediaRecorder.isTypeSupported('audio/webm'),
        opus: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    });
});

// =============================
// Recording Functions
// =============================

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    console.log("🎤 Starting recording...");
    
    const recordBtn = document.getElementById("recordBtn");
    const userInput = document.getElementById("userInput");
    
    try {
        recordBtn.disabled = true;
        userInput.textContent = "🎙️ Aufnahme läuft...";
        
        // Get microphone access
        currentStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                sampleRate: 16000
            }
        });
        
        // Setup MediaRecorder
        const options = {
            mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
                ? 'audio/webm;codecs=opus' 
                : 'audio/webm'
        };
        
        mediaRecorder = new MediaRecorder(currentStream, options);
        audioChunks = [];
        isRecording = true;
        
        // Event handlers
        mediaRecorder.ondataavailable = function(event) {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = processRecording;
        
        mediaRecorder.onerror = function(event) {
            console.error("❌ Recording error:", event.error);
            resetRecording("Aufnahmefehler");
        };
        
        // Start recording
        mediaRecorder.start(1000);
        recordBtn.textContent = "⏹️ Stop";
        
        // Auto-stop after 5 seconds
        setTimeout(function() {
            if (isRecording) {
                stopRecording();
            }
        }, 5000);
        
    } catch (error) {
        console.error("❌ Microphone error:", error);
        resetRecording("Mikrofon-Zugriff fehlgeschlagen");
        alert("Mikrofon-Fehler: " + error.message);
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        console.log("⏹️ Stopping recording...");
        mediaRecorder.stop();
        isRecording = false;
    }
}

async function processRecording() {
    console.log("🔄 Processing audio...");
    
    const userInput = document.getElementById("userInput");
    const botResponse = document.getElementById("botResponse");
    
    try {
        // Cleanup microphone
        if (currentStream) {
            currentStream.getTracks().forEach(function(track) {
                track.stop();
            });
            currentStream = null;
        }
        
        // Create audio blob
        const mimeType = mediaRecorder.mimeType || 'audio/webm';
        const audioBlob = new Blob(audioChunks, { type: mimeType });
        
        console.log("📝 Audio created:", audioBlob.size, "bytes,", mimeType);
        
        if (audioBlob.size < 1000) {
            throw new Error("Audio zu kurz oder leer");
        }
        
        // Step 1: Transcribe audio
        userInput.textContent = "⏳ Transkription...";
        const transcription = await transcribeAudio(audioBlob);
        
        console.log("📝 Transcription:", transcription);
        userInput.textContent = transcription;
        
        // Step 2: Generate response (always try if we have any text)
        if (transcription && transcription.trim().length > 0) {
            botResponse.textContent = "🧠 Denke nach...";
            
            try {
                const response = await generateResponse(transcription);
                console.log("🤖 Response:", response);
                botResponse.textContent = response;
                
                // Auto-play TTS if response is meaningful
                if (isGoodResponse(response)) {
                    setTimeout(function() {
                        playResponse();
                    }, 500);
                }
            } catch (error) {
                console.error("❌ Response generation failed:", error);
                botResponse.textContent = "Antwort-Generierung fehlgeschlagen";
            }
        } else {
            botResponse.textContent = "Bitte sprechen Sie deutlicher";
        }
        
    } catch (error) {
        console.error("❌ Processing failed:", error);
        userInput.textContent = "Fehler: " + error.message;
        botResponse.textContent = "Verarbeitung fehlgeschlagen";
    } finally {
        resetRecording();
    }
}

// =============================
// API Functions
// =============================

async function transcribeAudio(audioBlob) {
    const formData = new FormData();
    const fileName = audioBlob.type.includes('webm') ? 'voice.webm' : 'voice.wav';
    formData.append("file", audioBlob, fileName);
    
    const response = await fetch("http://localhost:8000/transcribe", {
        method: "POST",
        body: formData
    });
    
    if (!response.ok) {
        throw new Error("Transcription failed: " + response.status);
    }
    
    const data = await response.json();
    return data.text || "Keine Transkription";
}

async function generateResponse(text) {
    console.log("📤 Sending to response API:", text);
    
    const response = await fetch("http://localhost:8000/respond", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
    });
    
    console.log("📥 Response API status:", response.status);
    
    if (!response.ok) {
        throw new Error("Response generation failed: " + response.status);
    }
    
    const data = await response.json();
    console.log("📥 Response data:", data);
    
    return data.response || "Keine Antwort erhalten";
}

async function playResponse() {
    const text = document.getElementById("botResponse").textContent;
    const playBtn = document.getElementById("playBtn");
    
    console.log("🔊 Playing TTS for:", text);
    
    if (!isValidForTTS(text)) {
        alert("Kein gültiger Text zum Vorlesen");
        return;
    }
    
    try {
        playBtn.disabled = true;
        playBtn.textContent = "🔄 Lade...";
        
        const response = await fetch("http://localhost:8000/tts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error("TTS failed: " + response.status);
        }
        
        const audioBlob = await response.blob();
        
        if (audioBlob.size === 0) {
            throw new Error("Keine Audio-Daten erhalten");
        }
        
        // Play audio
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.onended = function() {
            URL.revokeObjectURL(audioUrl);
            console.log("✅ TTS playback finished");
        };
        
        audio.onerror = function() {
            throw new Error("Audio playback failed");
        };
        
        await audio.play();
        
    } catch (error) {
        console.error("❌ TTS error:", error);
        alert("TTS Fehler: " + error.message);
    } finally {
        playBtn.disabled = false;
        playBtn.textContent = "🔊 Antwort anhören";
    }
}

// =============================
// Utility Functions
// =============================

function isValidForTTS(text) {
    if (!text || typeof text !== 'string') return false;
    
    const cleaned = text.trim();
    
    // Must be long enough
    if (cleaned.length < 10) return false;
    
    // Exclude UI states and errors
    const invalidTexts = [
        '🧠', '⏳', '🔄', 'fehler', 'error', 'laden', 'denke nach',
        'verarbeitung', 'netzwerk', 'api', 'server', 'fehlgeschlagen'
    ];
    
    const lowerText = cleaned.toLowerCase();
    return !invalidTexts.some(function(invalid) {
        return lowerText.includes(invalid);
    });
}

function isGoodResponse(response) {
    return response && 
           response.length > 30 && 
           !response.toLowerCase().includes('entschuldigung') &&
           !response.toLowerCase().includes('keine information') &&
           !response.toLowerCase().includes('fehlgeschlagen');
}

function resetRecording(message) {
    const recordBtn = document.getElementById("recordBtn");
    
    recordBtn.disabled = false;
    recordBtn.textContent = "🎤 Start Aufnahme";
    isRecording = false;
    
    if (message) {
        document.getElementById("userInput").textContent = message;
    }
    
    // Cleanup
    if (currentStream) {
        currentStream.getTracks().forEach(function(track) {
            track.stop();
        });
        currentStream = null;
    }
}

function handleKeyPress(event) {
    // Spacebar: Stop recording
    if (event.code === "Space" && isRecording) {
        event.preventDefault();
        stopRecording();
    }
    
    // Enter: Play TTS
    if (event.code === "Enter" || event.code === "NumpadEnter") {
        const playBtn = document.getElementById("playBtn");
        if (playBtn && !playBtn.disabled) {
            playResponse();
        }
    }
}

async function checkBackendStatus() {
    try {
        const response = await fetch("http://localhost:8000/");
        
        if (response.ok) {
            const data = await response.json();
            console.log("✅ Backend online:", data);
            showStatus("✅ Online (" + (data.knowledge_entries || 0) + " KB entries)", 'green');
        } else {
            throw new Error("HTTP " + response.status);
        }
    } catch (error) {
        console.error("❌ Backend offline:", error);
        showStatus('❌ Backend Offline', 'red');
    }
}

function showStatus(message, color) {
    // Remove old status
    const oldStatus = document.getElementById('status-indicator');
    if (oldStatus) {
        oldStatus.remove();
    }
    
    // Create new status indicator
    const status = document.createElement('div');
    status.id = 'status-indicator';
    status.style.cssText = 
        'position: fixed; top: 10px; right: 10px; background: ' + color + '; ' +
        'color: white; padding: 8px 12px; border-radius: 6px; font-size: 12px; ' +
        'font-weight: bold; z-index: 1000; box-shadow: 0 2px 8px rgba(0,0,0,0.2);';
    status.textContent = message;
    
    document.body.appendChild(status);
}

function showSettings() {
    const info = 
        "VoiceBot Status:\n" +
        "- Browser: " + navigator.userAgent.split(' ')[0] + "\n" +
        "- Microphone: " + (currentStream ? 'Active' : 'Inactive') + "\n" +
        "- Audio Support: " + (MediaRecorder.isTypeSupported('audio/webm') ? 'WebM ✅' : 'WebM ❌') + "\n\n" +
        "Keyboard Shortcuts:\n" +
        "- Spacebar: Stop recording\n" +
        "- Enter: Play response\n\n" +
        "Backend: http://localhost:8000";
    
    alert(info);
}

// =============================
// Auto-Test Functions
// =============================

async function runAutoTest() {
    console.log("🧪 Running auto-test...");
    
    try {
        // Test response API
        console.log("🧪 Testing response API...");
        const testResponse = await fetch("http://localhost:8000/respond", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: "Test Karlsruhe" })
        });
        
        if (testResponse.ok) {
            const result = await testResponse.json();
            console.log("✅ Response API working:", result.response.substring(0, 50) + "...");
        } else {
            console.warn("⚠️ Response API failed:", testResponse.status);
        }
        
        // Test health endpoint
        console.log("🧪 Testing health endpoint...");
        const healthResponse = await fetch("http://localhost:8000/health");
        
        if (healthResponse.ok) {
            const health = await healthResponse.json();
            console.log("✅ Health check passed:", health.status);
        } else {
            console.warn("⚠️ Health check failed:", healthResponse.status);
        }
        
        console.log("✅ Auto-test completed");
        
    } catch (error) {
        console.error("❌ Auto-test failed:", error);
    }
}

// Run auto-test after initialization
setTimeout(function() {
    runAutoTest();
}, 2000);