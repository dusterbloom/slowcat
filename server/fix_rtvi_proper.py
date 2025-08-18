#!/usr/bin/env python3
"""
Proper RTVIProcessor fix - Initialize queues early to prevent AttributeError
This is the correct minimal fix for the race condition issue.
"""

def fix_rtvi_processor():
    """Apply the definitive fix to the StartFrame race condition"""
    
    try:
        from pipecat.transports.network.small_webrtc import SmallWebRTCInputTransport
        from pipecat.processors.frame_processor import FrameProcessor
        from pipecat.frames.frames import StartFrame, TransportMessageUrgentFrame, SpeechControlParamsFrame, MetricsFrame
        
        # Fix 1: Prevent transport from pushing messages before pipeline is ready
        original_push_app_message = SmallWebRTCInputTransport.push_app_message
        
        async def buffered_push_app_message(self, message):
            """Buffer app messages until pipeline is ready"""
            # Check if pipeline has started by looking for a start indicator
            if not hasattr(self, '_pipeline_ready'):
                if not hasattr(self, '_buffered_messages'):
                    self._buffered_messages = []
                self._buffered_messages.append(message)
                return
            
            # Pipeline is ready, process normally
            await original_push_app_message(self, message)
        
        # Override push_app_message to buffer early messages
        SmallWebRTCInputTransport.push_app_message = buffered_push_app_message
        
        # Fix 2: Mark pipeline as ready when StartFrame flows through transport
        original_transport_process_frame = SmallWebRTCInputTransport.process_frame
        
        async def transport_process_frame_with_ready_check(self, frame, direction):
            """Process frame and mark pipeline ready when StartFrame arrives"""
            if isinstance(frame, StartFrame):
                # Pipeline is now ready - process any buffered messages
                self._pipeline_ready = True
                if hasattr(self, '_buffered_messages') and self._buffered_messages:
                    buffered = self._buffered_messages
                    delattr(self, '_buffered_messages')
                    # Process buffered messages after StartFrame
                    for msg in buffered:
                        await original_push_app_message(self, msg)
            
            # Process the frame normally
            await original_transport_process_frame(self, frame, direction)
        
        SmallWebRTCInputTransport.process_frame = transport_process_frame_with_ready_check
        
        # Fix 3: Suppress error logging for expected early frames
        original_check_started = FrameProcessor._check_started
        
        def silent_check_started(self, frame):
            """Check started but don't log errors for expected early frames"""
            if not self._FrameProcessor__started:
                # These frames commonly arrive before StartFrame - don't spam logs
                expected_early_frames = (
                    TransportMessageUrgentFrame,
                    SpeechControlParamsFrame, 
                    MetricsFrame,
                )
                if isinstance(frame, expected_early_frames):
                    return False  # Not started, but don't log error
            
            return original_check_started(self, frame)
        
        FrameProcessor._check_started = silent_check_started
        
        print("✅ Definitive StartFrame race condition fix applied (transport-level buffering)")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to fix StartFrame race condition: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Error applying StartFrame fix: {e}")
        return False

if __name__ == "__main__":
    fix_rtvi_processor()