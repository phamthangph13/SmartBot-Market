#!/usr/bin/env python3
"""
Script to run the ZuneF.Com Chatbot
Usage: python run_chatbot.py
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def run_streamlit():
    """Run the Streamlit app"""
    print("Starting ZuneF.Com Chatbot...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "agent.py"])
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped by user")
    except Exception as e:
        print(f"❌ Error running chatbot: {e}")

def main():
    """Main function"""
    print("🚀 ZuneF.Com Chatbot Launcher")
    print("=" * 40)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Please create a .env file with your GEMINI_API_KEY")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Run the chatbot
    run_streamlit()

if __name__ == "__main__":
    main()
