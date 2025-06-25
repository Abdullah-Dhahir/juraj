#!/usr/bin/env python3
"""
Test script for VoiceBot services
Tests each component individually and the complete pipeline
"""

import asyncio
import logging
import tempfile
import wave
import numpy as np
import requests
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceBotTester:
    """Test suite for VoiceBot components"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_results = {}
    
    def create_test_audio(self, duration=2, sample_rate=16000):
        """Create a test audio file (sine wave)"""
        # Generate a simple sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # Convert to 16-bit integers
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            return temp_file.name
    
    def test_server_health(self):
        """Test if server is running and healthy"""
        logger.info("🔍 Testing server health...")
        
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Server is running: {data.get('message', 'OK')}")
                self.test_results['server_health'] = True
                return True
            else:
                logger.error(f"❌ Server health check failed: {response.status_code}")
                self.test_results['server_health'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Server not accessible: {e}")
            self.test_results['server_health'] = False
            return False
    
    def test_health_endpoint(self):
        """Test detailed health endpoint"""
        logger.info("🔍 Testing health endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Health endpoint working")
                
                # Check service status
                services = data.get('services', {})
                for service, info in services.items():
                    status = info.get('status', 'unknown')
                    logger.info(f"   {service}: {status}")
                
                self.test_results['health_endpoint'] = True
                return True
            else:
                logger.error(f"❌ Health endpoint failed: {response.status_code}")
                self.test_results['health_endpoint'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Health endpoint error: {e}")
            self.test_results['health_endpoint'] = False
            return False
    
    def test_stt_service(self):
        """Test Speech-to-Text service"""
        logger.info("🔍 Testing STT service...")
        
        try:
            # Create test audio
            audio_file = self.create_test_audio()
            
            # Test transcription endpoint
            with open(audio_file, 'rb') as f:
                files = {'file': ('test.wav', f, 'audio/wav')}
                response = requests.post(f"{self.base_url}/transcribe", files=files)
            
            # Clean up
            Path(audio_file).unlink()
            
            if response.status_code == 200:
                data = response.json()
                transcription = data.get('text', '')
                logger.info(f"✅ STT working, transcription: '{transcription}'")
                self.test_results['stt_service'] = True
                return True
            else:
                logger.error(f"❌ STT failed: {response.status_code} - {response.text}")
                self.test_results['stt_service'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ STT test error: {e}")
            self.test_results['stt_service'] = False
            return False
    
    def test_nlp_service(self):
        """Test NLP service"""
        logger.info("🔍 Testing NLP service...")
        
        test_messages = [
            "Hallo, wie geht es dir?",
            "Wo ist das Bürgerbüro in Karlsruhe?",
            "Hello, what are the opening hours?",
            "Wie kann ich mich anmelden?"
        ]
        
        try:
            for message in test_messages:
                payload = {"message": message}
                response = requests.post(
                    f"{self.base_url}/respond",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    bot_response = data.get('response', '')
                    logger.info(f"✅ NLP response to '{message[:20]}...': '{bot_response[:50]}...'")
                else:
                    logger.error(f"❌ NLP failed for '{message}': {response.status_code}")
                    self.test_results['nlp_service'] = False
                    return False
            
            logger.info("✅ NLP service working correctly")
            self.test_results['nlp_service'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ NLP test error: {e}")
            self.test_results['nlp_service'] = False
            return False
    
    def test_tts_service(self):
        """Test Text-to-Speech service"""
        logger.info("🔍 Testing TTS service...")
        
        test_texts = [
            "Hallo, das ist ein Test der deutschen Sprachausgabe.",
            "Hello, this is a test of English speech synthesis."
        ]
        
        try:
            for text in test_texts:
                payload = {"text": text}
                response = requests.post(
                    f"{self.base_url}/tts",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    # Check if we got audio data
                    content_type = response.headers.get('content-type', '')
                    if 'audio' in content_type:
                        logger.info(f"✅ TTS generated audio for: '{text[:30]}...'")
                    else:
                        logger.warning(f"⚠️ TTS returned non-audio content: {content_type}")
                else:
                    logger.error(f"❌ TTS failed for '{text[:20]}...': {response.status_code}")
                    self.test_results['tts_service'] = False
                    return False
            
            logger.info("✅ TTS service working correctly")
            self.test_results['tts_service'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ TTS test error: {e}")
            self.test_results['tts_service'] = False
            return False
    
    def test_complete_pipeline(self):
        """Test the complete voice bot pipeline"""
        logger.info("🔍 Testing complete pipeline...")
        
        try:
            # Step 1: Create test audio
            audio_file = self.create_test_audio()
            
            # Step 2: Transcribe audio
            with open(audio_file, 'rb') as f:
                files = {'file': ('test.wav', f, 'audio/wav')}
                stt_response = requests.post(f"{self.base_url}/transcribe", files=files)
            
            Path(audio_file).unlink()  # Clean up
            
            if stt_response.status_code != 200:
                logger.error("❌ Pipeline failed at STT step")
                self.test_results['complete_pipeline'] = False
                return False
            
            # Step 3: Generate response
            transcription = stt_response.json().get('text', 'Hallo')
            nlp_response = requests.post(
                f"{self.base_url}/respond",
                json={"message": transcription}
            )
            
            if nlp_response.status_code != 200:
                logger.error("❌ Pipeline failed at NLP step")
                self.test_results['complete_pipeline'] = False
                return False
            
            # Step 4: Generate speech
            bot_text = nlp_response.json().get('response', 'Antwort')
            tts_response = requests.post(
                f"{self.base_url}/tts",
                json={"text": bot_text}
            )
            
            if tts_response.status_code != 200:
                logger.error("❌ Pipeline failed at TTS step")
                self.test_results['complete_pipeline'] = False
                return False
            
            logger.info("✅ Complete pipeline working!")
            logger.info(f"   Input (simulated): [sine wave audio]")
            logger.info(f"   STT output: '{transcription}'")
            logger.info(f"   NLP output: '{bot_text[:50]}...'")
            logger.info(f"   TTS output: [audio file generated]")
            
            self.test_results['complete_pipeline'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Complete pipeline test error: {e}")
            self.test_results['complete_pipeline'] = False
            return False
    
    def run_all_tests(self):
        """Run all tests and provide summary"""
        logger.info("🧪 Starting VoiceBot test suite...")
        logger.info("=" * 50)
        
        tests = [
            ("Server Health", self.test_server_health),
            ("Health Endpoint", self.test_health_endpoint),
            ("STT Service", self.test_stt_service),
            ("NLP Service", self.test_nlp_service),
            ("TTS Service", self.test_tts_service),
            ("Complete Pipeline", self.test_complete_pipeline)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running: {test_name}")
            logger.info("-" * 30)
            
            try:
                if test_func():
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("🧪 TEST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Tests passed: {passed}/{total}")
        logger.info(f"Success rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 All tests passed! VoiceBot is working correctly.")
        else:
            logger.warning(f"⚠️ {total - passed} test(s) failed. Check the logs above.")
        
        return passed == total

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VoiceBot Test Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--test", choices=['health', 'stt', 'nlp', 'tts', 'pipeline'], help="Run specific test")
    
    args = parser.parse_args()
    
    tester = VoiceBotTester(args.url)
    
    if args.test:
        # Run specific test
        test_map = {
            'health': tester.test_server_health,
            'stt': tester.test_stt_service,
            'nlp': tester.test_nlp_service,
            'tts': tester.test_tts_service,
            'pipeline': tester.test_complete_pipeline
        }
        
        test_func = test_map.get(args.test)
        if test_func:
            success = test_func()
            exit(0 if success else 1)
    else:
        # Run all tests
        success = tester.run_all_tests()
        exit(0 if success else 1)

if __name__ == "__main__":
    main()