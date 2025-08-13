"use client";

import { useState, useEffect, useRef } from 'react';
import {
  ConsoleTemplate,
  FullScreenContainer,
} from "@pipecat-ai/voice-ui-kit";
import { 
  PipecatClientProvider,
  PipecatClientAudio 
} from '@pipecat-ai/client-react';
import { PipecatClient } from '@pipecat-ai/client-js';
import { SmallWebRTCTransport } from '@pipecat-ai/small-webrtc-transport';
import { VoiceReactivePlasma } from './VoiceReactivePlasma';
import { EnhancedExport } from './EnhancedExport';

interface VoiceAppProps {
  videoEnabled: boolean;
}

// Extend Window interface for TTS tracking
declare global {
  interface Window {
    currentTtsMessageId: string | null;
    ttsChunksReceived: Set<string> | null;
  }
}

type AppState = "idle" | "connecting" | "connected" | "disconnected";

// Type guard for checking if transcript has final property
const hasFinPropertyString = (obj: unknown): obj is { final: boolean } => {
  return typeof obj === 'object' && obj !== null && 'final' in obj;
};

// Text formatting functions
const formatUserText = (text: string): string => {
  // Fix capitalization and add basic punctuation
  if (!text) return text;
  
  // Convert to proper case (first letter of each sentence capitalized)
  let formatted = text.toLowerCase();
  
  // Capitalize first letter
  formatted = formatted.charAt(0).toUpperCase() + formatted.slice(1);
  
  // Capitalize after periods, exclamation marks, question marks
  formatted = formatted.replace(/[.!?]\s+\w/g, match => match.toUpperCase());
  
  // Add period at end if no punctuation exists
  if (!/[.!?]$/.test(formatted.trim())) {
    formatted += '.';
  }
  
  return formatted;
};

const formatBotText = (text: string): string => {
  // Remove markdown asterisks and other formatting
  if (!text) return text;
  
  // FIRST: Strip ALL emojis to match server-side sanitization
  let formatted = text
    // Basic emojis (U+1F600-U+1F64F)
    .replace(/[\u{1F600}-\u{1F64F}]/gu, '')
    // Miscellaneous symbols and pictographs (U+1F300-U+1F5FF)
    .replace(/[\u{1F300}-\u{1F5FF}]/gu, '')
    // Transport and map symbols (U+1F680-U+1F6FF)
    .replace(/[\u{1F680}-\u{1F6FF}]/gu, '')
    // Supplemental symbols (U+1F700-U+1F77F, U+1F780-U+1F7FF, U+1F800-U+1F8FF, U+1F900-U+1F9FF)
    .replace(/[\u{1F700}-\u{1F9FF}]/gu, '')
    // Additional emojis (U+1FA00-U+1FA6F, U+1FA70-U+1FAFF)
    .replace(/[\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}]/gu, '')
    // Enclosed alphanumeric supplement (U+1F100-U+1F1FF)
    .replace(/[\u{1F100}-\u{1F1FF}]/gu, '')
    // Miscellaneous symbols (U+2600-U+26FF)
    .replace(/[\u{2600}-\u{26FF}]/gu, '')
    // Dingbats (U+2700-U+27BF)
    .replace(/[\u{2700}-\u{27BF}]/gu, '')
    // Variation selectors (U+FE00-U+FE0F)
    .replace(/[\u{FE00}-\u{FE0F}]/gu, '')
    // Zero-width joiner and non-joiner
    .replace(/[\u200D\u200C]/g, '')
    // Common special symbols
    .replace(/[★☆♪♫♯♭♮⚡⭐🔥💯]/g, '');
  
  // Remove **bold** formatting
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '$1');
  
  // Remove *italic* formatting
  formatted = formatted.replace(/\*(.*?)\*/g, '$1');
  
  // Remove other common markdown
  formatted = formatted.replace(/`([^`]+)`/g, '$1'); // inline code
  formatted = formatted.replace(/^\s*[-*+]\s+/gm, '• '); // bullet points
  
  // Clean up any double spaces left from emoji removal
  formatted = formatted.replace(/\s+/g, ' ').trim();
  
  return formatted;
};

export function VoiceApp({ videoEnabled }: VoiceAppProps) {
  const [showDebugUI, setShowDebugUI] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isLowPowerMode, setIsLowPowerMode] = useState(false);
  const [client, setClient] = useState<PipecatClient | null>(null);
  const [appState, setAppState] = useState<AppState>("idle");
  const [isMicEnabled, setIsMicEnabled] = useState(true);
  const [isCameraEnabled, setIsCameraEnabled] = useState(false); // Always start with camera OFF for privacy
  const [showTranscript, setShowTranscript] = useState(false);
  const [transcriptExpanded, setTranscriptExpanded] = useState(false);
  const [fontSize, setFontSize] = useState('normal'); // 'small', 'normal', 'large'
  const [conversationHistory, setConversationHistory] = useState<Array<{role: 'user' | 'assistant', text: string, isStreaming?: boolean, id: string, timestamp?: Date}>>([]);
  const [currentUserTranscript, setCurrentUserTranscript] = useState('');
  const [currentAssistantTranscript, setCurrentAssistantTranscript] = useState('');
  const [currentTtsSegment, setCurrentTtsSegment] = useState(''); // Current TTS segment
  const currentTtsSegmentRef = useRef(''); // Ref for latest TTS segment
  const [llmMessageQueue, setLlmMessageQueue] = useState<string[]>([]); // Queue of LLM messages
  const llmMessageQueueRef = useRef<string[]>([]);
  // No need for complex index tracking - we just dequeue messages as they're spoken
  const [accumulatedTtsText, setAccumulatedTtsText] = useState(''); // Accumulated TTS text for current segment
  const accumulatedTtsTextRef = useRef<string>('');
  const lastSavedMessageRef = useRef<string>(''); // Track last saved message to prevent re-accumulation
  const [wasConnected, setWasConnected] = useState(false); // Track if we were ever connected
  const [autoReconnectAttempts, setAutoReconnectAttempts] = useState(0);
  const [showPerformanceMonitor, setShowPerformanceMonitor] = useState(false);
  const [performanceStats, setPerformanceStats] = useState({
    fps: 0,
    memoryUsage: 0,
    cpuTemp: 0,
    isWebGLActive: false
  });
  const [showVisualizerControls, setShowVisualizerControls] = useState(false);
  const [showExportPanel, setShowExportPanel] = useState(false);
  const transcriptScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initialize PipecatClient
    const initClient = async () => {
      const transport = new SmallWebRTCTransport();
      const pcClient = new PipecatClient({
        enableCam: false, // Start with camera disabled
        enableMic: true,  // Mic enabled by default
        transport: transport,
        callbacks: {
          // Debug: Log all bot events to see what we're getting
          onBotStartedSpeaking: () => console.log('🎙️ Bot started speaking'),
          onBotStoppedSpeaking: () => {
            console.log('🎙️ Bot stopped speaking - final check');
            // Final check for any remaining accumulated TTS when overall speaking stops
            const currentQueue = llmMessageQueueRef.current;
            const accumulatedTts = accumulatedTtsTextRef.current.trim();
            
            console.log('🔍 Final check - Queue length:', currentQueue.length, 'Accumulated:', accumulatedTts);
            
            // Final check using dequeue approach - check first message in queue
            if (currentQueue.length > 0 && accumulatedTts) {
              const expectedLlmMessage = currentQueue[0].trim();
              
              if (accumulatedTts === expectedLlmMessage) {
                // MATCH! Save the clean LLM message and dequeue it
                console.log('💾 Final save - matched first LLM message:', expectedLlmMessage.substring(0, 50) + '...');
                
                const messageId = `assistant-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
                setConversationHistory(prev => [
                  ...prev, 
                  { role: 'assistant', text: expectedLlmMessage, id: messageId, timestamp: new Date() }
                ]);
                
                // DEQUEUE: Remove the first message since it's been spoken and saved
                setLlmMessageQueue(prev => {
                  const updated = prev.slice(1);
                  llmMessageQueueRef.current = updated;
                  return updated;
                });
                
                console.log('📤 Final dequeue. Queue now has', currentQueue.length - 1, 'messages');
              } else {
                console.log('❌ Final check - no match:', `"${accumulatedTts}" vs "${expectedLlmMessage}"`);
              }
            }
            
            // Clear everything for next conversation
            setCurrentTtsSegment('');
            currentTtsSegmentRef.current = '';
            setCurrentAssistantTranscript('');
            setAccumulatedTtsText('');
            accumulatedTtsTextRef.current = '';
          },
          onTransportStateChanged: (state) => {
            switch (state) {
              case "connecting":
              case "authenticating":
                setAppState("connecting");
                break;
              case "ready":
              case "connected":
                setAppState("connected");
                setWasConnected(true);
                setAutoReconnectAttempts(0); // Reset attempts on successful connection
                break;
              case "disconnected":
              case "disconnecting":
                // If we were connected before, show disconnected state, otherwise idle
                setAppState(wasConnected ? "disconnected" : "idle");
                break;
              default:
                setAppState("idle");
                break;
            }
          },
          onError: (error) => {
            console.error("Pipecat error:", error);
            setAppState("disconnected");
          },
          onUserTranscript: (transcript) => {
            if (transcript && transcript.text) {
              // Handle both streaming (partial) and final transcripts
              // Check if transcript has 'final' property, otherwise assume final
              const isFinal = !hasFinPropertyString(transcript) || transcript.final !== false;
              
              if (!isFinal) {
                // Streaming/partial transcript - show in streaming bubble
                setCurrentUserTranscript(formatUserText(transcript.text));
              } else {
                // Final transcript - add to conversation history and clear streaming
                const messageId = `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                setConversationHistory(prev => [
                  ...prev, 
                  { 
                    role: 'user', 
                    text: formatUserText(transcript.text), 
                    id: messageId 
                  }
                ]);
                setCurrentUserTranscript('');
              }
            }
          },
          onBotTranscript: (transcript) => {
            // Add LLM message to queue - these are the clean versions we'll save
            if (transcript && transcript.text && transcript.text.trim().length > 0) {
              const cleanText = formatBotText(transcript.text.trim());
              
              setLlmMessageQueue(prev => {
                // Prevent duplicate messages from being added to the queue
                if (prev.length > 0 && prev[prev.length - 1] === cleanText) {
                  console.log('🚫 Duplicate LLM message ignored:', cleanText.substring(0, 50) + '...');
                  return prev; // Don't add duplicate
                }
                
                const updated = [...prev, cleanText];
                llmMessageQueueRef.current = updated;
                return updated;
              });
              console.log('📝 LLM message queued:', cleanText.substring(0, 50) + '...');
            }
          },
          // Handle TTS text with new protocol
          onBotTtsText: (ttsData) => {
            console.log('🎵 onBotTtsText:', ttsData);
            if (ttsData && ttsData.text && ttsData.text.trim().length > 0) {
              // Try to parse as JSON protocol first
              let messageData = null;
              let newText = '';
              
              try {
                const parsed = JSON.parse(ttsData.text);
                if (parsed.protocol === 'tts_v2') {
                  // New protocol with metadata
                  messageData = parsed;
                  newText = formatBotText(parsed.text).trim();
                  console.log(`📦 TTS Protocol v2 - Message: ${parsed.message_id}, Chunk: ${parsed.chunk_index}/${parsed.total_chunks}, Type: ${parsed.message_type}, Final: ${parsed.is_final}`);
                } else {
                  // Not our protocol, treat as plain text
                  newText = formatBotText(ttsData.text).trim();
                }
              } catch {
                // Not JSON, treat as plain text (fallback for old format)
                newText = formatBotText(ttsData.text).trim();
                console.log('📝 Plain text TTS (legacy mode):', newText);
              }
              
              const accumulated = (accumulatedTtsTextRef.current || '').trim();
              
              // Skip if empty text
              if (!newText) {
                console.log('⚠️ Empty text, skipping');
                return;
              }
              
              let updatedText = '';
              
              // Handle based on protocol version
              if (messageData && messageData.protocol === 'tts_v2') {
                // New protocol: Always incremental, properly tracked
                if (messageData.message_type === 'incremental') {
                  // Simply append incremental text
                  updatedText = accumulated ? (accumulated + ' ' + newText) : newText;
                  console.log(`➕ Appending incremental chunk ${messageData.chunk_index}: ${newText.substring(0, 30)}...`);
                } else if (messageData.message_type === 'cumulative') {
                  // Replace with cumulative text
                  updatedText = newText;
                  console.log(`🔄 Cumulative update chunk ${messageData.chunk_index}`);
                }
                
                // Store message ID for tracking
                if (!window.currentTtsMessageId || window.currentTtsMessageId !== messageData.message_id) {
                  console.log(`🆕 New TTS message started: ${messageData.message_id}`);
                  window.currentTtsMessageId = messageData.message_id;
                  window.ttsChunksReceived = new Set();
                }
                
                // Track received chunks to prevent duplicates
                const chunkKey = `${messageData.message_id}-${messageData.chunk_index}`;
                if (window.ttsChunksReceived?.has(chunkKey)) {
                  console.log(`🚫 Duplicate chunk ignored: ${chunkKey}`);
                  return;
                }
                window.ttsChunksReceived?.add(chunkKey);
                
              } else {
                // Legacy fallback logic (for old format)
                console.log('⚠️ Using legacy TTS handling');
                
                // Check if this text was already saved as a complete message
                if (lastSavedMessageRef.current === newText) {
                  console.log('🔄 This was already saved, ignoring:', newText);
                  return;
                }
                
                // Check if this is a duplicate or subset we already have
                if (accumulated === newText) {
                  console.log('🔁 Exact duplicate, ignoring');
                  return;
                }
                
                // Check if new text starts with what we have (cumulative update)
                if (accumulated && newText.startsWith(accumulated)) {
                  // Extract only the new part after what we already have
                  const newPart = newText.substring(accumulated.length).trim();
                  if (newPart) {
                    updatedText = accumulated + ' ' + newPart;
                    console.log('📊 Cumulative update, adding:', newPart);
                  } else {
                    console.log('🔁 No new content in cumulative update');
                    return;
                  }
                }
                // Check if we already have this EXACT text within our accumulated (not just substring)
                else if (accumulated && accumulated.split(' ').includes(newText)) {
                  console.log('🚫 Already have this exact word, ignoring:', newText);
                  return;
                }
                // Check if new text contains all our accumulated (full replacement)
                else if (accumulated && newText.includes(accumulated)) {
                  updatedText = newText;
                  console.log('🔄 Full replacement with larger text');
                }
                // New incremental text to append
                else {
                  updatedText = accumulated ? (accumulated + ' ' + newText) : newText;
                  console.log('➕ Appending new text:', newText);
                }
              }
              
              // Store the updated text (already trimmed)
              updatedText = updatedText.trim();
              console.log('📚 Updated total:', updatedText.substring(0, 60) + '...');
              
              setAccumulatedTtsText(updatedText);
              accumulatedTtsTextRef.current = updatedText;
              setCurrentAssistantTranscript(updatedText);
              
              // Check for message completion
              const currentQueue = llmMessageQueueRef.current;
              let isMessageComplete = false;
              
              // Check if this is the final chunk in the new protocol
              if (messageData && messageData.is_final) {
                console.log(`🏁 Final chunk received for message ${messageData.message_id}`);
                isMessageComplete = true;
              }
              
              // Also check for perfect match with LLM queue (for both protocols)
              if (currentQueue.length > 0) {
                const expectedMessage = currentQueue[0].trim(); // Trim expected message too
                
                if (updatedText === expectedMessage) {
                  console.log('✅ Perfect match with LLM queue!');
                  isMessageComplete = true;
                }
              }
              
              // Save complete message
              if (isMessageComplete && updatedText) {
                console.log('💾 Saving complete message to transcript');
                
                // Track what we saved to prevent re-accumulation
                lastSavedMessageRef.current = updatedText;
                
                const messageId = `assistant-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
                setConversationHistory(prev => [
                  ...prev, 
                  { role: 'assistant', text: updatedText, id: messageId, timestamp: new Date() }
                ]);
                
                // Remove from queue if we matched
                if (currentQueue.length > 0 && updatedText === currentQueue[0].trim()) {
                  setLlmMessageQueue(prev => {
                    const updated = prev.slice(1);
                    llmMessageQueueRef.current = updated;
                    return updated;
                  });
                  console.log('🎯 Dequeued message. Queue remaining:', currentQueue.length - 1);
                }
                
                // Clear for next message
                setAccumulatedTtsText('');
                accumulatedTtsTextRef.current = '';
                setCurrentAssistantTranscript('');
                
                // Clear message tracking
                if (window.currentTtsMessageId) {
                  console.log(`✨ Message ${window.currentTtsMessageId} complete and saved`);
                  window.currentTtsMessageId = null;
                  window.ttsChunksReceived = null;
                }
              }
            }
          },
          onBotTtsStarted: () => {
            console.log('🎤 Bot TTS started');
            // Don't clear accumulated text here - we're building it incrementally
            // Only clear the display if we're starting a completely new message
            if (!accumulatedTtsTextRef.current) {
              setCurrentTtsSegment('');
              currentTtsSegmentRef.current = '';
              setCurrentAssistantTranscript('');
            }
          },
          onBotTtsStopped: () => {
            console.log('Bot TTS stopped - checking for complete message');
            
            const currentQueue = llmMessageQueueRef.current;
            const accumulatedTts = accumulatedTtsTextRef.current.trim();
            
            console.log('🔍 TTS stopped - Queue length:', currentQueue.length, 'Accumulated:', accumulatedTts);
            
            // Check if accumulated text matches any message in queue
            if (currentQueue.length > 0 && accumulatedTts) {
              const firstMessage = currentQueue[0].trim();
              
              // Only save and clear if we have a COMPLETE match
              if (accumulatedTts === firstMessage) {
                console.log('✅ TTS stopped with complete message - saving:', firstMessage.substring(0, 50) + '...');
                
                // Track and save - DISABLED to prevent duplicates with new protocol
                lastSavedMessageRef.current = firstMessage;
                
                // DISABLED: Saving is now handled by is_final flag in onBotTtsText
                // const messageId = `assistant-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
                // setConversationHistory(prev => [
                //   ...prev, 
                //   { role: 'assistant', text: firstMessage, id: messageId, timestamp: new Date() }
                // ]);
                
                // Dequeue the message
                setLlmMessageQueue(prev => {
                  const updated = prev.slice(1);
                  llmMessageQueueRef.current = updated;
                  return updated;
                });
                
                // Clear for next message since this one is complete
                setCurrentTtsSegment('');
                currentTtsSegmentRef.current = '';
                setCurrentAssistantTranscript('');
                setAccumulatedTtsText('');
                accumulatedTtsTextRef.current = '';
                
                console.log('📤 Complete message saved. Queue now has', currentQueue.length - 1, 'messages');
              } else if (firstMessage.includes(accumulatedTts)) {
                // The accumulated text is found within the expected message - it's a partial match
                // This handles cases with emojis or when TTS stops mid-sentence
                console.log('⚠️ TTS stopped mid-sentence. Keeping accumulated text for continuation.');
                // DON'T clear the accumulator - keep building on it
                // Just clear the display but keep the accumulator intact
                setCurrentTtsSegment('');
                currentTtsSegmentRef.current = '';
              } else {
                console.log('❌ TTS stopped with unmatched text:', accumulatedTts);
                console.log('Expected to find it in:', firstMessage.substring(0, 60) + '...');
                // Only clear if there's no match at all
                setCurrentTtsSegment('');
                currentTtsSegmentRef.current = '';
                setCurrentAssistantTranscript('');
                setAccumulatedTtsText('');
                accumulatedTtsTextRef.current = '';
              }
            } else {
              // No queue or no accumulated text - safe to clear
              setCurrentTtsSegment('');
              currentTtsSegmentRef.current = '';
              setCurrentAssistantTranscript('');
              if (!currentQueue.length) {
                // Only clear accumulator if queue is empty
                setAccumulatedTtsText('');
                accumulatedTtsTextRef.current = '';
              }
            }
          },
        },
      });
      
      await pcClient.initDevices();
      setClient(pcClient);
    };

    initClient();
  }, []);

  // Auto-scroll transcript to bottom when new messages arrive
  useEffect(() => {
    if (transcriptScrollRef.current && showTranscript && conversationHistory.length > 0) {
      const scrollElement = transcriptScrollRef.current;
      scrollElement.scrollTop = scrollElement.scrollHeight;
    }
  }, [conversationHistory, showTranscript]);

  // Auto-reconnection logic for kings 👑
  useEffect(() => {
    if (appState === "disconnected" && wasConnected && autoReconnectAttempts < 5) {
      const delay = Math.min(1000 * Math.pow(2, autoReconnectAttempts), 10000); // Exponential backoff, max 10s
      console.log(`🔄 Auto-reconnecting in ${delay}ms (attempt ${autoReconnectAttempts + 1}/5)...`);
      
      const reconnectTimeout = setTimeout(async () => {
        if (appState === "disconnected") { // Still disconnected
          setAutoReconnectAttempts(prev => prev + 1);
          try {
            // Ensure clean disconnect before reconnecting
            if (client) {
              client.disconnect();
              // Small delay to ensure cleanup
              await new Promise(resolve => setTimeout(resolve, 100));
            }
            // Clear conversation state for fresh start
            setConversationHistory([]);
            setCurrentAssistantTranscript('');
            setCurrentTtsSegment('');
            setCurrentUserTranscript('');
            setLlmMessageQueue([]);
            llmMessageQueueRef.current = [];
            setAccumulatedTtsText('');
            accumulatedTtsTextRef.current = '';
            await handleConnect();
          } catch (error) {
            console.error("Auto-reconnect failed:", error);
          }
        }
      }, delay);

      return () => clearTimeout(reconnectTimeout);
    }
  }, [appState, wasConnected, autoReconnectAttempts]);

  // Performance monitoring
  useEffect(() => {
    if (!showPerformanceMonitor) return;

    let frameCount = 0;
    let lastTime = performance.now();
    let monitoring = true;

    const updatePerformance = () => {
      frameCount++;
      const now = performance.now();
      
      // Update FPS every second
      if (now - lastTime >= 1000) {
        const fps = Math.round((frameCount * 1000) / (now - lastTime));
        
        // Get memory usage (if available)
        const memoryInfo = (performance as typeof performance & { memory?: { usedJSHeapSize: number } }).memory;
        const memoryUsage = memoryInfo ? Math.round(memoryInfo.usedJSHeapSize / 1048576) : 0; // MB
        
        // Simply check if we're NOT in low power mode to determine if WebGL should be active
        // Don't try to access canvas context as it causes conflicts with THREE.js
        const isWebGLActive = !isLowPowerMode && !!document.querySelector('canvas');

        setPerformanceStats({
          fps,
          memoryUsage,
          cpuTemp: 0, // Browser can't access CPU temp directly
          isWebGLActive
        });

        frameCount = 0;
        lastTime = now;
      }

      if (monitoring) {
        requestAnimationFrame(updatePerformance);
      }
    };

    requestAnimationFrame(updatePerformance);

    return () => {
      monitoring = false;
    };
  }, [showPerformanceMonitor]);

  const handleConnect = async () => {
    if (!client || (appState !== "idle" && appState !== "disconnected")) return;
    
    try {
      await client.connect({
        connectionUrl: "/api/offer",
      });
    } catch (error) {
      console.error("Connection error:", error);
      setAppState("disconnected");
    }
  };

  const handleDisconnect = () => {
    if (!client) return;
    client.disconnect();
    setConversationHistory([]);
    setCurrentAssistantTranscript('');
    setCurrentTtsSegment('');
    setLlmMessageQueue([]);
    llmMessageQueueRef.current = [];
    setAccumulatedTtsText('');
    accumulatedTtsTextRef.current = '';
  };

  const toggleMicrophone = async () => {
    if (!client) return;
    
    try {
      if (isMicEnabled) {
        client.enableMic(false);
      } else {
        client.enableMic(true);
      }
      setIsMicEnabled(!isMicEnabled);
    } catch (error) {
      console.error('Error toggling microphone:', error);
    }
  };

  const toggleCamera = async () => {
    if (!client) return;
    
    try {
      if (isCameraEnabled) {
        client.enableCam(false);
      } else {
        client.enableCam(true);
      }
      setIsCameraEnabled(!isCameraEnabled);
    } catch (error) {
      console.error('Error toggling camera:', error);
    }
  };

  const exportConversation = () => {
    const conversation = conversationHistory.map(msg => 
      `**${msg.role === 'user' ? 'User' : 'Assistant'}:** ${msg.text}`
    ).join('\n\n');
    const blob = new Blob([conversation], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `slowcat-conversation-${new Date().toISOString().slice(0, 19)}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (showDebugUI && client) {
    // Debug/dev mode - use shared client to preserve connection
    return (
      <PipecatClientProvider client={client}>
        <FullScreenContainer>
          <div className="relative w-full h-full">
            {/* Debug mode toggle button */}
            <button
              onClick={() => setShowDebugUI(false)}
              className="absolute top-2 right-40 z-50 bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm"
            >
              Enhanced UI
            </button>
            
            <ConsoleTemplate
              transportType="smallwebrtc"
              connectParams={{
                connectionUrl: "/api/offer",
              }}
              noUserVideo={!videoEnabled}
            />
          </div>
        </FullScreenContainer>
        <PipecatClientAudio />
      </PipecatClientProvider>
    );
  }

  // Enhanced UX mode with PlasmaVisualizer
  return (
    <PipecatClientProvider client={client!}>
      <FullScreenContainer className={isDarkMode ? 'bg-black' : 'bg-white'}>
        <div 
          className={`relative w-full h-full group ${isDarkMode ? 'bg-black' : 'bg-white'}`}
        >
          {/* Invisible interaction areas for controls */}
          <div className="absolute inset-0 z-50">
            
            {/* Connect/Disconnect in center when needed - ALWAYS VISIBLE IN IDLE */}
            {appState === "idle" && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <button
                    onClick={handleConnect}
                    className={`
                      relative px-12 py-4 rounded-full transition-all duration-300
                      ${isDarkMode 
                        ? 'bg-white text-black hover:bg-gray-100' 
                        : 'bg-black text-white hover:bg-gray-900'}
                      hover:scale-[1.02] active:scale-[0.98] transform
                    `}
                  >
                    <span className="font-light text-base tracking-[0.2em] uppercase">
                      Connect
                    </span>
                  </button>
                </div>
              </div>
            )}

            {appState === "connecting" && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className={`
                    animate-spin w-10 h-10 rounded-full mx-auto mb-4
                    ${isDarkMode 
                      ? 'border-2 border-white/20 border-t-white/60' 
                      : 'border-2 border-black/20 border-t-black/60'}
                  `}></div>
                  <p className={`
                    font-light text-xs tracking-[0.15em] uppercase
                    ${isDarkMode ? 'text-white/60' : 'text-black/60'}
                  `}>
                    Connecting
                  </p>
                </div>
              </div>
            )}


            {/* Always visible controls - no hover required */}
            {/* Top-left: Connection Status
            <div className="absolute top-6 left-6">
              <div className={`
                px-4 py-2 rounded-full backdrop-blur-sm transition-all duration-500
                ${isDarkMode 
                  ? 'bg-transparent border border-black/30' 
                  : 'bg-transparent border border-white/30'}
              `}>
                <span className={`
                  font-light text-xs tracking-[0.15em] uppercase
                  ${isDarkMode ? 'text-black' : 'text-white'}
                `}>
                  {appState === 'connected' ? '● Connected' : 
                   appState === 'connecting' ? '◐ Connecting' : 
                   appState === 'disconnected' && autoReconnectAttempts > 0 ? `🔄 Reconnecting ${autoReconnectAttempts}/5` :
                   appState === 'disconnected' ? '⚡ Disconnected' :
                   '○ Offline'}
                </span>
              </div>
            </div> */}

            {/* Top-right: Control buttons - always visible */}
            <div className="absolute top-6 right-6 flex gap-3">
              <button
                onClick={() => setIsDarkMode(!isDarkMode)}
                className={`
                  w-10 h-10 rounded-full transition-all duration-500 flex items-center justify-center
                  ${isDarkMode 
                    ? 'bg-white/90 hover:bg-white text-black' 
                    : 'bg-black/90 hover:bg-black text-white'}
                  hover:scale-[1.05] active:scale-[0.95] transform
                `}
              >
                <span className="text-sm">{isDarkMode ? '☀' : '☽'}</span>
              </button>
              <button
                onClick={() => setShowPerformanceMonitor(!showPerformanceMonitor)}
                className={`
                  px-4 py-2 rounded-full transition-all duration-500 flex items-center gap-2
                  ${isDarkMode 
                    ? (showPerformanceMonitor ? 'bg-white text-black' : 'bg-white/90 hover:bg-white text-black')
                    : (showPerformanceMonitor ? 'bg-black text-white' : 'bg-black/90 hover:bg-black text-white')}
                  hover:scale-[1.02] active:scale-[0.98] transform
                `}
                title="Toggle performance monitor"
              >
                <span className="text-sm">📊</span>
                <span className="font-light text-xs tracking-[0.15em] uppercase">
                  Perf
                </span>
              </button>
              {isLowPowerMode && (
                <button
                  onClick={() => setShowVisualizerControls(!showVisualizerControls)}
                  className={`
                    w-10 h-10 rounded-full transition-all duration-500 flex items-center justify-center
                    ${isDarkMode 
                      ? (showVisualizerControls ? 'bg-white text-black' : 'bg-white/90 hover:bg-white text-black')
                      : (showVisualizerControls ? 'bg-black text-white' : 'bg-black/90 hover:bg-black text-white')}
                    hover:scale-[1.05] active:scale-[0.95] transform
                  `}
                  title="Visualizer settings"
                >
                  <span className="text-sm">⚙️</span>
                </button>
              )}
              <button
                onClick={() => setIsLowPowerMode(!isLowPowerMode)}
                className={`
                  px-4 py-2 rounded-full transition-all duration-500 flex items-center gap-2
                  ${isDarkMode 
                    ? (isLowPowerMode ? 'bg-white text-black' : 'bg-white/90 hover:bg-white text-black')
                    : (isLowPowerMode ? 'bg-black text-white' : 'bg-black/90 hover:bg-black text-white')}
                  hover:scale-[1.02] active:scale-[0.98] transform
                `}
                title={isLowPowerMode ? 'Disable low power mode' : 'Enable low power mode'}
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24" className="w-3 h-3">
                  <path d="M16 4h-1V2h-6v2H8C6.9 4 6 4.9 6 6v14c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2z"/>
                </svg>
                <span className="font-light text-xs tracking-[0.15em] uppercase">
                  {isLowPowerMode ? 'Low Power' : 'Normal'}
                </span>
              </button>
              <button
                onClick={() => setShowExportPanel(!showExportPanel)}
                className={`
                  px-4 py-2 rounded-full transition-all duration-500 flex items-center gap-2
                  ${isDarkMode 
                    ? (showExportPanel ? 'bg-green-600 text-white' : 'bg-white/90 hover:bg-white text-black')
                    : (showExportPanel ? 'bg-green-500 text-white' : 'bg-black/90 hover:bg-black text-white')}
                  hover:scale-[1.02] active:scale-[0.98] transform
                `}
              >
                <span className="text-sm">📥</span>
                <span className="font-light text-xs tracking-[0.15em] uppercase">
                  {showExportPanel ? 'Hide Export' : 'Export'}
                </span>
              </button>
              <button
                onClick={() => setShowDebugUI(true)}
                className={`
                  px-4 py-2 rounded-full transition-all duration-500
                  ${isDarkMode 
                    ? 'bg-white/90 hover:bg-white text-black' 
                    : 'bg-black/90 hover:bg-black text-white'}
                  hover:scale-[1.02] active:scale-[0.98] transform
                `}
              >
                <span className="font-light text-xs tracking-[0.15em] uppercase">Debug</span>
              </button>
            </div>

            {/* Bottom-left: Audio/Video controls - ALWAYS VISIBLE (even when disconnected) */}
            {(appState === "connected" || appState === "disconnected") && (
              <div className="absolute bottom-6 left-6 flex gap-3">
                <button
                  onClick={toggleMicrophone}
                  disabled={appState !== "connected"}
                  className={`
                    px-4 py-2 rounded-full transition-all duration-300 flex items-center gap-2
                    ${appState !== "connected" ? 'opacity-50 cursor-not-allowed' : ''}
                    ${isDarkMode 
                      ? (isMicEnabled ? 'bg-white/90 text-black' : 'bg-white/50 text-black/50') 
                      : (isMicEnabled ? 'bg-black/90 text-white' : 'bg-black/50 text-white/50')}
                    ${appState === "connected" ? 'hover:scale-[1.02] active:scale-[0.98] transform' : ''}
                  `}
                >
                  <span className="text-sm">
                    {appState !== "connected" ? '🔌' : (isMicEnabled ? '🎤' : '🔇')}
                  </span>
                  <span className="font-light text-xs tracking-[0.15em] uppercase">
                    {appState !== "connected" ? 'Disconnected' : (isMicEnabled ? 'Mic On' : 'Mic Off')}
                  </span>
                </button>
                
                <button
                  onClick={toggleCamera}
                  disabled={appState !== "connected"}
                  className={`
                    px-4 py-2 rounded-full transition-all duration-300 flex items-center gap-2
                    ${appState !== "connected" ? 'opacity-50 cursor-not-allowed' : ''}
                    ${isDarkMode 
                      ? (isCameraEnabled ? 'bg-white/90 text-black' : 'bg-white/50 text-black/50') 
                      : (isCameraEnabled ? 'bg-black/90 text-white' : 'bg-black/50 text-white/50')}
                    ${appState === "connected" ? 'hover:scale-[1.02] active:scale-[0.98] transform' : ''}
                  `}
                >
                  <span className="text-sm">
                    {appState !== "connected" ? '🔌' : (isCameraEnabled ? '📹' : '📷')}
                  </span>
                  <span className="font-light text-xs tracking-[0.15em] uppercase">
                    {appState !== "connected" ? 'Disconnected' : (isCameraEnabled ? 'Camera On' : 'Camera Off')}
                  </span>
                </button>
              </div>
            )}

            {/* Bottom-center: Disconnect/Reconnect button */}
            {(appState === "connected" || appState === "disconnected") && (
              <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2">
                <button
                  onClick={appState === "connected" ? handleDisconnect : async () => {
                    try {
                      // Manual reconnect - ensure clean state first
                      if (client) {
                        client.disconnect();
                        await new Promise(resolve => setTimeout(resolve, 100));
                      }
                      setAutoReconnectAttempts(0); // Reset auto attempts when manually connecting
                      setConversationHistory([]); // Clear conversation history
                      setCurrentAssistantTranscript(''); // Clear current assistant transcript
                      setCurrentTtsSegment(''); // Clear current TTS segment
                      setCurrentUserTranscript(''); // Clear current user transcript
                      setLlmMessageQueue([]); // Clear LLM message queue
                      llmMessageQueueRef.current = [];
                      setAccumulatedTtsText(''); // Clear accumulated TTS
                      accumulatedTtsTextRef.current = '';
                      await handleConnect();
                    } catch (error) {
                      console.error("Manual reconnect failed:", error);
                    }
                  }}
                  className={`
                    px-4 py-2 rounded-full transition-all duration-300
                    ${isDarkMode 
                      ? 'bg-white/90 text-black hover:bg-white' 
                      : 'bg-black/90 text-white hover:bg-black'}
                    hover:scale-[1.02] active:scale-[0.98] transform
                  `}
                >
                  <span className="font-light text-xs tracking-[0.15em] uppercase">
                    {appState === "connected" ? 'Disconnect' : 'Reconnect'}
                  </span>
                </button>
              </div>
            )}
            
            {/* Bottom-right: Transcript toggle - always visible when connected */}
            {appState === "connected" && (
              <div className="absolute bottom-6 right-6">
                <button
                  onClick={() => setShowTranscript(!showTranscript)}
                  className={`
                    px-4 py-2 rounded-full transition-all duration-300
                    ${isDarkMode 
                      ? (showTranscript ? 'bg-white text-black' : 'bg-white/90 text-black')
                      : (showTranscript ? 'bg-black text-white' : 'bg-black/90 text-white')}
                    hover:scale-[1.02] active:scale-[0.98] transform
                  `}
                >
                  <span className="font-light text-xs tracking-[0.15em] uppercase">
                    {showTranscript ? 'Hide' : 'Show'} Transcript
                  </span>
                </button>
              </div>
            )}
          </div>

          {/* Voice-reactive PlasmaVisualizer background - PURE, NO OVERLAYS */}
          <VoiceReactivePlasma 
            isDarkMode={isDarkMode} 
            isLowPowerMode={isLowPowerMode} 
            showControls={showVisualizerControls}
          />

          {/* Performance Monitor Overlay */}
          {showPerformanceMonitor && (
            <div className="absolute top-20 left-6 z-50">
              <div className={`
                px-4 py-3 rounded-lg backdrop-blur-sm transition-all duration-500 text-xs
                ${isDarkMode 
                  ? 'bg-black/80 border border-white/20 text-white' 
                  : 'bg-white/80 border border-black/20 text-black'}
              `}>
                <div className="font-bold tracking-wide uppercase mb-2">⚡ Performance Monitor</div>
                <div className="space-y-1 font-mono">
                  <div className="flex justify-between gap-4">
                    <span>FPS:</span>
                    <span className={`font-bold ${
                      performanceStats.fps >= 30 ? 'text-green-500' : 
                      performanceStats.fps >= 15 ? 'text-yellow-500' : 'text-red-500'
                    }`}>
                      {performanceStats.fps}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span>Memory:</span>
                    <span className={`font-bold ${
                      performanceStats.memoryUsage < 100 ? 'text-green-500' : 
                      performanceStats.memoryUsage < 200 ? 'text-yellow-500' : 'text-red-500'
                    }`}>
                      {performanceStats.memoryUsage}MB
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span>WebGL:</span>
                    <span className={`font-bold ${performanceStats.isWebGLActive ? 'text-blue-500' : 'text-gray-500'}`}>
                      {performanceStats.isWebGLActive ? 'ACTIVE' : 'OFF'}
                    </span>
                  </div>
                  <div className="border-t border-opacity-20 pt-2 mt-2">
                    <div className="text-xs opacity-70">
                      {performanceStats.fps < 15 && performanceStats.isWebGLActive && 
                        '⚠️ Low FPS detected - consider Low Power mode'}
                      {performanceStats.memoryUsage > 200 && 
                        '⚠️ High memory usage detected'}
                      {!performanceStats.isWebGLActive && 
                        '✅ Using fallback renderer'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>


        {/* Transcript Panel - Full width with transparent background */}
        {showTranscript && (
          <div className={`
            absolute transition-all duration-500 z-50
            ${transcriptExpanded 
              ? 'bottom-0 left-0 right-0 top-32' 
              : 'bottom-20 left-0 right-0'}
             ${isDarkMode ? 'bg-white/70' : 'bg-black/70'}
               backdrop-blur-sm
          `}>
            {/* Header */}
            <div className="flex justify-between items-center px-6 py-3">
              <h3 className={`
                font-bold tracking-[0.2em] uppercase
                ${isDarkMode ?  'text-black' : 'text-white'}
                ${fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-sm' : 'text-xs'}
              `}>
                Conversation
              </h3>
              <div className="flex gap-3">
                <button
                  onClick={() => setTranscriptExpanded(!transcriptExpanded)}
                  className={`
                    text-xs transition-all duration-300
                    ${isDarkMode 
                      ? 'text-black/60 hover:text-black'
                      : 'text-white/60 hover:text-white' }
                  `}
                >
                  {transcriptExpanded ? '▼' : '▲'}
                </button>
                <button
                  onClick={() => {
                    const sizes = ['small', 'normal', 'large'];
                    const currentIndex = sizes.indexOf(fontSize);
                    setFontSize(sizes[(currentIndex + 1) % 3]);
                  }}
                  className={`
                    text-xs transition-all duration-300
                    ${isDarkMode 
                      ? 'text-black/60 hover:text-black'
                      : 'text-white/60 hover:text-white' }
                  `}
                >
                  A{fontSize === 'large' ? '+' : fontSize === 'small' ? '-' : ''}
                </button>
                <button
                  onClick={exportConversation}
                  className={`
                    text-xs transition-all duration-300
                    ${isDarkMode 
                      ? 'text-black/60 hover:text-black'
                      : 'text-white/60 hover:text-white' }
                  `}
                >
                  Export
                </button>
                <button
                  onClick={() => setShowTranscript(false)}
                  className={`
                    text-xs transition-all duration-300
                    ${isDarkMode 
                      ? 'text-black/60 hover:text-black'
                      : 'text-white/60 hover:text-white' }
                  `}
                >
                  ✕
                </button>
              </div>
            </div>
            
            {/* Transcript Content */}
            <div 
              ref={transcriptScrollRef}
              className={`
                px-6 pb-3 overflow-y-auto
                ${transcriptExpanded ? 'max-h-[calc(100vh-12rem)]' : 'max-h-24'}
              `}
              style={{ scrollBehavior: 'smooth' }}
            >
              {/* Show conversation history if available */}
              {conversationHistory.length > 0 ? (
                <div className="space-y-3">
                  {conversationHistory.map((message, idx) => (
                    <div key={message.id || idx} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} mb-3`}>
                      <div className={`max-w-[80%] rounded-lg p-3 ${
                        message.role === 'user' 
                          ? 'bg-black text-white' 
                          : (isDarkMode ? 'bg-gray-700 text-gray-100 border border-gray-600' : 'bg-gray-200 text-gray-900 border border-gray-300')
                      }`}>
                        <div 
                          className={`leading-relaxed break-words ${
                            fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-base' : 'text-sm'
                          }`}
                        >
                          {message.text}
                        </div>
                        <div className={`text-xs mt-1 opacity-70 ${
                          message.role === 'user' ? 'text-white' : (isDarkMode ? 'text-gray-400' : 'text-gray-500')
                        }`}>
                          {message.timestamp ? message.timestamp.toLocaleTimeString([], { 
                            hour: '2-digit', 
                            minute: '2-digit' 
                          }) : new Date().toLocaleTimeString([], { 
                            hour: '2-digit', 
                            minute: '2-digit' 
                          })}
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {/* Streaming user transcript */}
                  {currentUserTranscript && (
                    <div className="flex justify-end mb-3">
                      <div className="max-w-[80%] rounded-lg p-3 bg-black text-white opacity-90">
                        <div className={`leading-relaxed break-words ${
                          fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-base' : 'text-sm'
                        }`}>
                          {currentUserTranscript}
                        </div>
                        <div className="text-xs mt-1 opacity-70 text-white flex items-center gap-1">
                          <span>Speaking...</span>
                          <div className="flex gap-1">
                            <div className="w-1 h-1 bg-white rounded-full animate-pulse"></div>
                            <div className="w-1 h-1 bg-white rounded-full animate-pulse" style={{ animationDelay: '0.1s' }}></div>
                            <div className="w-1 h-1 bg-white rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* Streaming assistant transcript */}
                  {currentAssistantTranscript && (
                    <div className="flex justify-start mb-3">
                      <div className={`max-w-[80%] rounded-lg p-3 ${
                        isDarkMode ? 'bg-gray-700 text-gray-100 border border-gray-600' : 'bg-gray-200 text-gray-900 border border-gray-300'
                      } opacity-90`}>
                        <div className={`leading-relaxed break-words ${
                          fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-base' : 'text-sm'
                        }`}>
                          {currentAssistantTranscript}
                        </div>
                        <div className={`text-xs mt-1 opacity-70 flex items-center gap-1 ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>
                          <span>Assistant...</span>
                          <div className="flex gap-1">
                            <div className={`w-1 h-1 rounded-full animate-pulse ${
                              isDarkMode ? 'bg-gray-400' : 'bg-gray-500'
                            }`}></div>
                            <div className={`w-1 h-1 rounded-full animate-pulse ${
                              isDarkMode ? 'bg-gray-400' : 'bg-gray-500'
                            }`} style={{ animationDelay: '0.1s' }}></div>
                            <div className={`w-1 h-1 rounded-full animate-pulse ${
                              isDarkMode ? 'bg-gray-400' : 'bg-gray-500'
                            }`} style={{ animationDelay: '0.2s' }}></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                </div>
              ) : (
                <div className={`text-center py-8 ${
                  isDarkMode ? 'text-white/60' : 'text-black/60'
                }`}>
                  <p className="text-sm">Start a conversation to see the transcript here</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Enhanced Export Panel - Slides in from right */}
        {showExportPanel && (
          <div className={`absolute top-0 right-0 bottom-0 w-96 z-[60] transition-all duration-500 ${
            isDarkMode ? 'bg-black/90' : 'bg-white/90'
          } backdrop-blur-sm border-l ${
            isDarkMode ? 'border-white/20' : 'border-black/20'
          }`}>
            <EnhancedExport
              conversationHistory={conversationHistory}
              isDarkMode={isDarkMode}
              className="h-full"
            />
            <button
              onClick={() => setShowExportPanel(false)}
              className={`absolute top-4 right-4 w-8 h-8 rounded-full transition-all duration-300 flex items-center justify-center ${
                isDarkMode
                  ? 'bg-white/20 hover:bg-white/30 text-white'
                  : 'bg-black/20 hover:bg-black/30 text-black'
              }`}
            >
              ✕
            </button>
          </div>
        )}

      </FullScreenContainer>
      
      {/* Audio component */}
      <PipecatClientAudio />
    </PipecatClientProvider>
  );
}