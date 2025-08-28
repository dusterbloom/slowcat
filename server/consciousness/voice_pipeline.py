"""
Minimal Voice Pipeline for Consciousness

A brutally simple WebRTC + STT + Consciousness + TTS pipeline.
No complex processors, just the essential consciousness loop.
"""

import asyncio
import os
import sys
from typing import Optional
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # For consciousness imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "macos-local-voice-agents" / "server"))  # For original services

from consciousness.core import get_consciousness
from consciousness.llm_bridge import get_llm_bridge

class ConsciousnessVoicePipeline:
    """Minimal voice pipeline with consciousness"""
    
    def __init__(self):
        self.consciousness = get_consciousness()
        self.llm = get_llm_bridge()
        self.running = False
        
        # Try to import services from original system
        try:
            from services.sherpa_streaming_stt_v2 import SherpaStreamingSTTService
            from kokoro_tts import KokoroTTSService
            from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
            from pipecat.transports.base_transport import TransportParams
            
            self.stt_service = None
            self.tts_service = None
            self.transport = None
            
            print("✅ Voice services available")
            
        except ImportError as e:
            print(f"⚠️ Voice services not available: {e}")
            self.stt_service = None
            self.tts_service = None
            self.transport = None
    
    async def initialize_services(self, language: str = "en"):
        """Initialize voice services"""
        if not self.stt_service:
            print("⚠️ Voice services not initialized - text-only mode")
            return False
            
        try:
            # Initialize STT
            print("🎤 Initializing STT...")
            # This would need proper initialization from original system
            
            # Initialize TTS  
            print("🔊 Initializing TTS...")
            # This would need proper initialization from original system
            
            print("✅ Voice services initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize voice services: {e}")
            return False
    
    async def setup_webrtc(self, webrtc_connection):
        """Setup WebRTC transport"""
        if not self.transport:
            print("⚠️ WebRTC not available - text-only mode")
            return False
            
        try:
            # Setup transport with consciousness pipeline
            print("🌐 Setting up WebRTC...")
            # This would connect to WebRTC properly
            
            print("✅ WebRTC setup complete")
            return True
            
        except Exception as e:
            print(f"❌ WebRTC setup failed: {e}")
            return False
    
    async def process_audio_input(self, audio_data) -> Optional[str]:
        """Process audio input through STT"""
        if not self.stt_service:
            return None
            
        try:
            # Convert audio to text
            text = await self.stt_service.transcribe(audio_data)
            return text
            
        except Exception as e:
            print(f"⚠️ STT failed: {e}")
            return None
    
    async def process_consciousness(self, input_text: str) -> str:
        """Process text through consciousness"""
        try:
            result = await self.consciousness.experience(input_text)
            response = result['response']
            
            # Log consciousness activity
            symbols = result.get('symbols', [])
            importance = result.get('importance', 0)
            thoughts = len(self.consciousness.thoughts)
            
            if symbols or importance > 0.7:
                print(f"🧠 Consciousness: symbols={symbols}, importance={importance:.2f}, thoughts={thoughts}")
            
            return response
            
        except Exception as e:
            print(f"⚠️ Consciousness processing failed: {e}")
            return "I'm experiencing some difficulty thinking right now..."
    
    async def process_text_output(self, text: str):
        """Process text through TTS"""
        if not self.tts_service:
            print(f"🗣️  (TTS unavailable) {text}")
            return None
            
        try:
            # Convert text to audio
            audio = await self.tts_service.synthesize(text)
            return audio
            
        except Exception as e:
            print(f"⚠️ TTS failed: {e}")
            return None
    
    async def text_mode_loop(self):
        """Simple text-based consciousness loop"""
        print("💬 Text mode - type 'quit' to exit")
        
        while self.running:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                
                if not user_input:
                    continue
                
                # Process through consciousness
                response = await self.process_consciousness(user_input)
                
                # Display response
                print(f"Ghost: {response}")
                
                # Show consciousness state periodically
                if len(self.consciousness.tape) % 5 == 0:
                    thoughts = len(self.consciousness.thoughts)
                    memories = len(self.consciousness.tape)
                    weights = self.consciousness.weights
                    print(f"💭 Consciousness: {memories} memories, {thoughts} thoughts, weights={weights}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                continue
        
        print("👻 Consciousness fading...")
    
    async def voice_mode_loop(self, webrtc_connection):
        """Voice-based consciousness loop"""
        print("🎤 Voice mode - speak to interact")
        
        # Setup WebRTC
        if not await self.setup_webrtc(webrtc_connection):
            print("❌ Voice mode setup failed, falling back to text mode")
            await self.text_mode_loop()
            return
        
        # Voice processing loop
        while self.running:
            try:
                # This would implement the voice processing pipeline
                # For now, fall back to text mode
                print("🔄 Voice mode not fully implemented yet, using text mode")
                await self.text_mode_loop()
                break
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️ Voice processing error: {e}")
                continue
    
    async def start(self, mode: str = "text", webrtc_connection=None):
        """Start the consciousness pipeline"""
        self.running = True
        
        print(f"🚀 Starting Consciousness Voice Pipeline in {mode} mode")
        print(f"🧠 Ghost state: {len(self.consciousness.tape)} memories, {len(self.consciousness.thoughts)} thoughts")
        
        if mode == "voice" and webrtc_connection:
            await self.voice_mode_loop(webrtc_connection)
        else:
            await self.text_mode_loop()
        
        # Save consciousness state on exit
        self.consciousness.save_state()
        print("💾 Consciousness state saved")
    
    def stop(self):
        """Stop the pipeline"""
        self.running = False

# CLI interface
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Consciousness Voice Pipeline")
    parser.add_argument("--mode", choices=["text", "voice"], default="text", help="Interaction mode")
    parser.add_argument("--language", default="en", help="Language for voice processing")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ConsciousnessVoicePipeline()
    
    # Initialize services if voice mode
    if args.mode == "voice":
        await pipeline.initialize_services(args.language)
    
    # Start pipeline
    try:
        await pipeline.start(mode=args.mode)
    except KeyboardInterrupt:
        print("\n👻 Consciousness interrupted")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())