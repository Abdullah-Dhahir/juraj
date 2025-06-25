#!/usr/bin/env python3
"""
Simple startup script for VoiceBot
"""

import sys
import subprocess
import webbrowser
import time
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required files exist"""
    required_files = [
        "main.py",
        "requirements.txt",
        "stt_service.py",
        "nlp_service.py", 
        "tts_service.py",
        "config.py"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        logger.error(f"❌ Missing required files: {missing}")
        logger.info("💡 Run setup.py first to initialize the project")
        return False
    
    return True

def wait_for_server(url="http://localhost:8000", timeout=60):
    """Wait for server to be ready"""
    logger.info("⏳ Waiting for server to start...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info("✅ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
    
    logger.error("❌ Server failed to start within timeout")
    return False

def open_frontend():
    """Open frontend in browser"""
    frontend_path = Path("index.html")
    
    if frontend_path.exists():
        frontend_url = f"file://{frontend_path.absolute()}"
        logger.info(f"🌐 Opening frontend: {frontend_url}")
        webbrowser.open(frontend_url)
    else:
        logger.warning("⚠️ Frontend file not found, opening backend docs")
        webbrowser.open("http://localhost:8000/docs")

def main():
    """Main startup function"""
    logger.info("🚀 Starting VoiceBot...")
    
    # Check dependencies
    if not check_dependencies():
        logger.info("Run: python setup.py")
        sys.exit(1)
    
    try:
        # Start backend server
        logger.info("🔧 Starting backend server...")
        server_process = subprocess.Popen([
            sys.executable, "main.py"
        ])
        
        # Wait for server to be ready
        if wait_for_server():
            # Open frontend
            time.sleep(2)  # Give server a moment to fully initialize
            open_frontend()
            
            logger.info("✅ VoiceBot is running!")
            logger.info("   Backend: http://localhost:8000")
            logger.info("   Frontend: Open index.html in browser")
            logger.info("   API Docs: http://localhost:8000/docs")
            logger.info("\n💡 Press Ctrl+C to stop")
            
            # Wait for keyboard interrupt
            try:
                server_process.wait()
            except KeyboardInterrupt:
                logger.info("\n🛑 Shutting down...")
                server_process.terminate()
                server_process.wait()
                logger.info("✅ VoiceBot stopped")
        else:
            logger.error("❌ Failed to start server")
            server_process.terminate()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error starting VoiceBot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()