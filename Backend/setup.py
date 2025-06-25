#!/usr/bin/env python3
"""
VoiceBot Setup Script
Downloads models and sets up the environment
"""

import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path
from typing import Dict, Any
import logging

from config import settings, MODEL_URLS, ensure_directories

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceBotSetup:
    """Setup class for VoiceBot installation"""
    
    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        self.data_dir = Path("data")
        
    def check_python_version(self):
        """Check Python version compatibility"""
        logger.info("🐍 Checking Python version...")
        
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8+ required")
            return False
        
        logger.info(f"✅ Python {sys.version} is compatible")
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("📦 Installing dependencies...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            logger.info("✅ Dependencies installed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """Download a specific model"""
        if model_name not in MODEL_URLS:
            logger.error(f"❌ Unknown model: {model_name}")
            return False
        
        model_info = MODEL_URLS[model_name]
        model_path = self.models_dir / model_info["filename"]
        
        # Check if model already exists
        if model_path.exists() and not force:
            logger.info(f"✅ Model {model_name} already exists")
            return True
        
        logger.info(f"📥 Downloading {model_name} ({model_info['size']})...")
        logger.info(f"    URL: {model_info['url']}")
        logger.info(f"    Destination: {model_path}")
        
        try:
            # Create models directory
            self.models_dir.mkdir(exist_ok=True, parents=True)
            
            # Download with progress
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, (downloaded * 100) // total_size)
                    print(f"\r    Progress: {percent}%", end='', flush=True)
            
            urllib.request.urlretrieve(
                model_info["url"], 
                model_path, 
                reporthook=progress_hook
            )
            
            print()  # New line after progress
            logger.info(f"✅ {model_name} downloaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {e}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            return False
    
    def setup_knowledge_base(self):
        """Setup initial knowledge base"""
        logger.info("📚 Setting up knowledge base...")
        
        knowledge_base_path = self.data_dir / "knowledge_base.json"
        
        if knowledge_base_path.exists():
            logger.info("✅ Knowledge base already exists")
            return True
        
        # Create sample knowledge base
        import json
        from nlp_service import NLPService
        
        # Get sample data
        nlp_service = NLPService.__new__(NLPService)  # Don't initialize
        sample_data = nlp_service._get_karlsruhe_knowledge()
        
        try:
            with open(knowledge_base_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
            logger.info("✅ Knowledge base created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create knowledge base: {e}")
            return False
    
    def create_env_file(self):
        """Create .env file with default settings"""
        env_path = Path(".env")
        
        if env_path.exists():
            logger.info("✅ .env file already exists")
            return True
        
        logger.info("⚙️ Creating .env file...")
        
        env_content = f"""# VoiceBot Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Model Settings
WHISPER_MODEL_SIZE=base
USE_COQUI_TTS=true
LLM_MODEL_PATH={self.models_dir}/mistral-7b-instruct-v0.1.Q4_K_M.gguf

# Performance
USE_GPU=true
MAX_CONCURRENT_REQUESTS=5

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/voicebot.log
"""
        
        try:
            with open(env_path, 'w') as f:
                f.write(env_content)
            
            logger.info("✅ .env file created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create .env file: {e}")
            return False
    
    def check_gpu_support(self):
        """Check for GPU support"""
        logger.info("🖥️ Checking GPU support...")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✅ GPU available: {gpu_name} ({gpu_count} device(s))")
                return True
            else:
                logger.info("💻 No GPU available, will use CPU")
                return False
                
        except ImportError:
            logger.info("⚠️ PyTorch not installed yet, GPU check skipped")
            return False
    
    def run_full_setup(self, download_models: bool = True):
        """Run complete setup process"""
        logger.info("🚀 Starting VoiceBot setup...")
        
        steps = [
            ("Python version", self.check_python_version),
            ("Directories", lambda: ensure_directories() or True),
            ("Environment file", self.create_env_file),
            ("Dependencies", self.install_dependencies),
            ("GPU support", self.check_gpu_support),
            ("Knowledge base", self.setup_knowledge_base)
        ]
        
        # Add model download steps if requested
        if download_models:
            steps.append(("Mistral model", lambda: self.download_model("mistral-7b-instruct")))
        
        failed_steps = []
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"Step: {step_name}")
            logger.info(f"{'='*50}")
            
            try:
                success = step_func()
                if not success:
                    failed_steps.append(step_name)
                    logger.error(f"❌ Step failed: {step_name}")
                else:
                    logger.info(f"✅ Step completed: {step_name}")
                    
            except Exception as e:
                failed_steps.append(step_name)
                logger.error(f"❌ Step failed with exception: {step_name} - {e}")
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("Setup Summary")
        logger.info(f"{'='*50}")
        
        if failed_steps:
            logger.error(f"❌ Setup completed with {len(failed_steps)} failed steps:")
            for step in failed_steps:
                logger.error(f"   - {step}")
        else:
            logger.info("✅ Setup completed successfully!")
            logger.info("\nTo start the VoiceBot server, run:")
            logger.info("   python main.py")
        
        return len(failed_steps) == 0

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VoiceBot Setup Script")
    parser.add_argument("--no-models", action="store_true", help="Skip model downloads")
    parser.add_argument("--model", type=str, help="Download specific model only")
    parser.add_argument("--force", action="store_true", help="Force re-download of models")
    
    args = parser.parse_args()
    
    setup = VoiceBotSetup()
    
    if args.model:
        # Download specific model only
        success = setup.download_model(args.model, force=args.force)
        sys.exit(0 if success else 1)
    else:
        # Full setup
        success = setup.run_full_setup(download_models=not args.no_models)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()