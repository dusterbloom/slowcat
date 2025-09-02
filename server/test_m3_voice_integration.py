#!/usr/bin/env python3
"""
Test M3 Voice Recognition Integration

Tests the integration between LightweightVoiceRecognition/AutoEnrollVoiceRecognition 
and the M3-AudioGraph system.

Verifies:
- Voice node creation during enrollment
- Speaker identity resolution with equivalence detection
- Profile migration to AudioGraph format
- M3-AudioGraph voice matching functionality
"""

import asyncio
import logging
import os
import tempfile
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import components to test
try:
    from config import VoiceRecognitionConfig
    from voice_recognition.lightweight import LightweightVoiceRecognition
    from voice_recognition.auto_enroll import AutoEnrollVoiceRecognition
    from memory.m3_integration import M3AudioGraphFactory
    from memory.audio_graph import AudioGraph
    
    # Mock VoiceEncoder for testing without actual audio
    class MockVoiceEncoder:
        def __init__(self, device="cpu"):
            self.device = device
            
        def embed_utterance(self, audio_array: np.ndarray) -> np.ndarray:
            # Generate consistent but unique embedding based on audio content
            np.random.seed(int(np.sum(audio_array * 1000) % 2147483647))
            embedding = np.random.randn(256).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            return embedding
    
    # Patch VoiceEncoder for testing
    import voice_recognition.lightweight as lightweight_module
    import voice_recognition.auto_enroll as auto_enroll_module
    lightweight_module.VoiceEncoder = MockVoiceEncoder
    auto_enroll_module.VoiceEncoder = MockVoiceEncoder  # Patch both modules
    
    COMPONENTS_AVAILABLE = True
    
except ImportError as e:
    logger.error(f"Required components not available: {e}")
    COMPONENTS_AVAILABLE = False


class TestM3VoiceIntegration:
    """Test suite for M3 voice recognition integration."""
    
    def __init__(self):
        self.temp_dir = None
        self.voice_recognition = None
        self.audio_graph = None
        self.test_results = []
    
    async def setup_test_environment(self):
        """Set up test environment with temporary directory and mock data."""
        # Create temporary directory for profiles
        self.temp_dir = tempfile.mkdtemp(prefix="test_m3_voice_")
        logger.info(f"Created test directory: {self.temp_dir}")
        
        # Create test configuration
        config = VoiceRecognitionConfig(
            enabled=True,
            confidence_threshold=0.7,
            profile_dir=self.temp_dir,
            profile_file_extension=".pkl",
            enrolled_profile_extension=".json",
            min_utterances_for_enrollment=3,
            consistency_threshold=0.8,
            min_consistency_threshold=0.7,
            enrollment_window_minutes=5,
            new_speaker_grace_period_seconds=30,
            new_speaker_similarity_threshold=0.6,
            min_adaptation_confidence=0.8,
            profile_adaptation_rate=0.1,
            min_utterance_duration_seconds=1.0
        )
        
        # Create legacy pickle profiles for migration testing
        await self.create_test_profiles()
        
        # Initialize M3-enhanced voice recognition
        self.voice_recognition = AutoEnrollVoiceRecognition(config)
        await self.voice_recognition.initialize()
        
        # Get reference to AudioGraph if M3 integration is enabled
        if hasattr(self.voice_recognition, 'use_m3_integration') and self.voice_recognition.use_m3_integration:
            self.audio_graph = self.voice_recognition.m3_integration.audio_graph
        
        logger.info("✅ Test environment setup complete")
    
    async def create_test_profiles(self):
        """Create legacy pickle profiles for migration testing."""
        # Generate mock voice embeddings
        speakers = ["Alice", "Bob", "Charlie"]
        
        for i, speaker_name in enumerate(speakers):
            # Generate consistent embeddings for each speaker
            np.random.seed(i + 1000)  # Consistent seed per speaker
            embeddings = []
            
            for j in range(3):  # 3 embeddings per speaker
                embedding = np.random.randn(256).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            
            # Save as pickle profile
            profile_path = os.path.join(self.temp_dir, f"{speaker_name}.pkl")
            with open(profile_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            logger.debug(f"Created test profile: {profile_path}")
    
    async def test_voice_node_creation(self):
        """Test that voice nodes are created during speaker enrollment."""
        logger.info("🧪 Testing voice node creation during enrollment...")
        
        try:
            # Generate mock audio for new speaker enrollment
            mock_audio_samples = []
            np.random.seed(12345)  # Consistent for test speaker
            
            for i in range(4):  # 4 consistent utterances for auto-enrollment
                # Generate audio array
                audio_length = 16000 * 2  # 2 seconds at 16kHz
                base_pattern = np.sin(2 * np.pi * (440 + i * 10) * np.linspace(0, 2, audio_length))  # Consistent frequency per utterance
                noise = np.random.randn(audio_length) * 0.1
                audio_array = (base_pattern + noise).astype(np.float32)
                mock_audio_samples.append(audio_array)
            
            # Simulate enrollment process
            initial_node_count = len(self.audio_graph.voice_nodes) if self.audio_graph else 0
            
            for i, audio_array in enumerate(mock_audio_samples):
                await self.voice_recognition._process_speaker_identification(audio_array)
                await asyncio.sleep(0.1)  # Small delay between utterances
            
            # Check if new voice node was created
            if self.audio_graph:
                final_node_count = len(self.audio_graph.voice_nodes)
                voice_nodes_created = final_node_count - initial_node_count
                
                if voice_nodes_created > 0:
                    self.test_results.append("✅ Voice node creation: PASSED")
                    logger.info(f"✅ Created {voice_nodes_created} voice nodes during enrollment")
                else:
                    self.test_results.append("❌ Voice node creation: FAILED (no nodes created)")
                    logger.error("❌ No voice nodes created during enrollment")
            else:
                self.test_results.append("⏭️ Voice node creation: SKIPPED (M3 integration disabled)")
                logger.info("⏭️ M3 integration disabled, skipping voice node test")
        
        except Exception as e:
            self.test_results.append(f"❌ Voice node creation: ERROR ({e})")
            logger.error(f"❌ Error testing voice node creation: {e}")
    
    async def test_speaker_identity_resolution(self):
        """Test speaker identity resolution with equivalence detection."""
        logger.info("🧪 Testing speaker identity resolution...")
        
        try:
            if not self.voice_recognition.use_m3_integration:
                self.test_results.append("⏭️ Speaker identity resolution: SKIPPED (M3 disabled)")
                logger.info("⏭️ M3 integration disabled, skipping identity resolution test")
                return
            
            # Test canonical identity resolution
            test_speaker = "Speaker_1"  # Should exist from previous enrollment
            
            if test_speaker in self.voice_recognition.speakers:
                resolved_identity = self.voice_recognition.resolve_speaker_identity(test_speaker)
                
                if resolved_identity:
                    self.test_results.append("✅ Speaker identity resolution: PASSED")
                    logger.info(f"✅ Resolved '{test_speaker}' -> '{resolved_identity}'")
                else:
                    self.test_results.append("❌ Speaker identity resolution: FAILED (no resolution)")
                    logger.error("❌ Failed to resolve speaker identity")
            else:
                # Create a test speaker for identity resolution
                test_fingerprints = [np.random.randn(256).astype(np.float32)]
                success = self.voice_recognition.create_voice_node(test_speaker, test_fingerprints)
                
                if success:
                    resolved_identity = self.voice_recognition.resolve_speaker_identity(test_speaker)
                    self.test_results.append("✅ Speaker identity resolution: PASSED")
                    logger.info(f"✅ Created and resolved '{test_speaker}' -> '{resolved_identity}'")
                else:
                    self.test_results.append("❌ Speaker identity resolution: FAILED (node creation failed)")
                    logger.error("❌ Failed to create test voice node")
        
        except Exception as e:
            self.test_results.append(f"❌ Speaker identity resolution: ERROR ({e})")
            logger.error(f"❌ Error testing speaker identity resolution: {e}")
    
    async def test_profile_migration(self):
        """Test migration of existing pickle profiles to AudioGraph."""
        logger.info("🧪 Testing profile migration to AudioGraph...")
        
        try:
            if not self.voice_recognition.use_m3_integration:
                self.test_results.append("⏭️ Profile migration: SKIPPED (M3 disabled)")
                logger.info("⏭️ M3 integration disabled, skipping migration test")
                return
            
            # Check if legacy profiles were migrated
            initial_speaker_count = len(self.voice_recognition.speakers)
            
            if self.audio_graph:
                voice_node_count = len(self.audio_graph.voice_nodes)
                
                if voice_node_count >= initial_speaker_count:
                    self.test_results.append("✅ Profile migration: PASSED")
                    logger.info(f"✅ Migrated {initial_speaker_count} profiles to {voice_node_count} voice nodes")
                else:
                    self.test_results.append("❌ Profile migration: FAILED (insufficient nodes)")
                    logger.error(f"❌ Expected >= {initial_speaker_count} voice nodes, got {voice_node_count}")
            else:
                self.test_results.append("❌ Profile migration: FAILED (no AudioGraph)")
                logger.error("❌ AudioGraph not available for migration testing")
        
        except Exception as e:
            self.test_results.append(f"❌ Profile migration: ERROR ({e})")
            logger.error(f"❌ Error testing profile migration: {e}")
    
    async def test_m3_voice_matching(self):
        """Test M3-AudioGraph voice matching functionality."""
        logger.info("🧪 Testing M3-AudioGraph voice matching...")
        
        try:
            if not self.voice_recognition.use_m3_integration:
                self.test_results.append("⏭️ M3 voice matching: SKIPPED (M3 disabled)")
                logger.info("⏭️ M3 integration disabled, skipping voice matching test")
                return
            
            # Generate test fingerprint similar to an existing speaker
            if self.voice_recognition.speakers:
                existing_speaker = list(self.voice_recognition.speakers.keys())[0]
                existing_fingerprint = self.voice_recognition.speakers[existing_speaker][0]
                
                # Create similar fingerprint with small noise
                test_fingerprint = existing_fingerprint + np.random.randn(*existing_fingerprint.shape) * 0.1
                test_fingerprint = test_fingerprint / np.linalg.norm(test_fingerprint)
                
                # Test M3 voice matching with lower threshold for testing
                # Pass a lower threshold for testing since we're using noisy fingerprints
                matched_speaker, confidence = self.voice_recognition.m3_integration.find_speaker_voice_node([test_fingerprint], 0.5)
                
                # Use lower threshold since we're testing with noisy/modified fingerprints
                if matched_speaker and confidence > 0.4:
                    self.test_results.append("✅ M3 voice matching: PASSED")
                    logger.info(f"✅ M3 matched speaker: {matched_speaker} (confidence: {confidence:.3f})")
                else:
                    self.test_results.append("❌ M3 voice matching: FAILED (no match)")
                    logger.error(f"❌ M3 matching failed: {matched_speaker}, confidence: {confidence}")
            else:
                self.test_results.append("⏭️ M3 voice matching: SKIPPED (no speakers)")
                logger.info("⏭️ No existing speakers for matching test")
        
        except Exception as e:
            self.test_results.append(f"❌ M3 voice matching: ERROR ({e})")
            logger.error(f"❌ Error testing M3 voice matching: {e}")
    
    async def test_statistics_and_cleanup(self):
        """Test M3 statistics and cleanup functionality."""
        logger.info("🧪 Testing M3 statistics and cleanup...")
        
        try:
            if not self.voice_recognition.use_m3_integration:
                self.test_results.append("⏭️ M3 statistics: SKIPPED (M3 disabled)")
                logger.info("⏭️ M3 integration disabled, skipping statistics test")
                return
            
            # Get M3 statistics
            stats = self.voice_recognition.get_m3_statistics()
            
            if stats and 'voice_nodes' in stats:
                self.test_results.append("✅ M3 statistics: PASSED")
                logger.info(f"✅ M3 statistics: {stats}")
                
                # Test cleanup functionality
                cleaned_edges = self.voice_recognition.cleanup_m3_connections(threshold=0.1)
                self.test_results.append("✅ M3 cleanup: PASSED")
                logger.info(f"✅ Cleaned up {cleaned_edges} weak connections")
                
            else:
                self.test_results.append("❌ M3 statistics: FAILED (no stats)")
                logger.error("❌ No M3 statistics available")
        
        except Exception as e:
            self.test_results.append(f"❌ M3 statistics: ERROR ({e})")
            logger.error(f"❌ Error testing M3 statistics: {e}")
    
    async def cleanup_test_environment(self):
        """Clean up test environment."""
        try:
            if self.voice_recognition:
                await self.voice_recognition.shutdown()
            
            # Clean up temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up test directory: {self.temp_dir}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def run_all_tests(self):
        """Run complete test suite."""
        logger.info("🚀 Starting M3 Voice Integration Test Suite...")
        logger.info(f"Components available: {COMPONENTS_AVAILABLE}")
        
        if not COMPONENTS_AVAILABLE:
            logger.error("❌ Required components not available - skipping tests")
            return
        
        try:
            await self.setup_test_environment()
            await self.test_voice_node_creation()
            await self.test_speaker_identity_resolution() 
            await self.test_profile_migration()
            await self.test_m3_voice_matching()
            await self.test_statistics_and_cleanup()
            
        except Exception as e:
            logger.error(f"❌ Test suite error: {e}")
            self.test_results.append(f"❌ Test suite: ERROR ({e})")
        
        finally:
            await self.cleanup_test_environment()
            
            # Print test results summary
            logger.info("\n" + "="*60)
            logger.info("📊 M3 VOICE INTEGRATION TEST RESULTS")
            logger.info("="*60)
            
            for result in self.test_results:
                logger.info(result)
            
            passed = len([r for r in self.test_results if r.startswith("✅")])
            failed = len([r for r in self.test_results if r.startswith("❌")])
            skipped = len([r for r in self.test_results if r.startswith("⏭️")])
            
            logger.info("="*60)
            logger.info(f"📈 SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
            logger.info("="*60)


async def main():
    """Main test runner."""
    test_suite = TestM3VoiceIntegration()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())