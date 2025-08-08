"use client";

import { useState, useEffect } from 'react';
import {
  ConsoleTemplate,
  FullScreenContainer,
  TranscriptOverlay,
} from "@pipecat-ai/voice-ui-kit";
import { 
  PipecatClientProvider,
  PipecatClientAudio 
} from '@pipecat-ai/client-react';
import { PipecatClient } from '@pipecat-ai/client-js';
import { SmallWebRTCTransport } from '@pipecat-ai/small-webrtc-transport';
import { VoiceReactivePlasma } from './VoiceReactivePlasma';

interface VoiceAppProps {
  videoEnabled: boolean;
}

type AppState = "idle" | "connecting" | "connected" | "disconnected";

export function VoiceApp({ videoEnabled }: VoiceAppProps) {
  const [showDebugUI, setShowDebugUI] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [client, setClient] = useState<PipecatClient | null>(null);
  const [appState, setAppState] = useState<AppState>("idle");
  const [isMicEnabled, setIsMicEnabled] = useState(true);
  const [isCameraEnabled, setIsCameraEnabled] = useState(false); // Always start with camera OFF for privacy
  const [showTranscript, setShowTranscript] = useState(false);
  const [transcriptExpanded, setTranscriptExpanded] = useState(false);
  const [fontSize, setFontSize] = useState('normal'); // 'small', 'normal', 'large'
  const [conversationHistory, setConversationHistory] = useState<Array<{role: 'user' | 'assistant', text: string}>>([]);
  const [showControls, setShowControls] = useState(true);

  useEffect(() => {
    // Initialize PipecatClient
    const initClient = async () => {
      const transport = new SmallWebRTCTransport();
      const pcClient = new PipecatClient({
        enableCam: false, // Start with camera disabled
        enableMic: true,  // Mic enabled by default
        transport: transport,
        callbacks: {
          onTransportStateChanged: (state) => {
            switch (state) {
              case "connecting":
              case "authenticating":
                setAppState("connecting");
                break;
              case "ready":
              case "connected":
                setAppState("connected");
                break;
              case "disconnected":
              case "disconnecting":
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
              setConversationHistory(prev => [...prev, { role: 'user', text: transcript.text }]);
            }
          },
          onBotTranscript: (transcript) => {
            if (transcript && transcript.text) {
              setConversationHistory(prev => [...prev, { role: 'assistant', text: transcript.text }]);
            }
          },
        },
      });
      
      await pcClient.initDevices();
      setClient(pcClient);
    };

    initClient();
  }, []);

  const handleConnect = async () => {
    if (!client || appState !== "idle") return;
    
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
          onMouseEnter={() => setShowControls(true)}
          onMouseLeave={() => setShowControls(false)}
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
            {/* Top-left: Connection Status */}
            <div className="absolute top-6 left-6">
              <div className={`
                px-4 py-2 rounded-full backdrop-blur-sm transition-all duration-500
                ${isDarkMode 
                  ? 'bg-transparent border border-white/30' 
                  : 'bg-transparent border border-black/30'}
              `}>
                <span className={`
                  font-light text-xs tracking-[0.15em] uppercase
                  ${isDarkMode ? 'text-white' : 'text-black'}
                `}>
                  {appState === 'connected' ? '● Connected' : 
                   appState === 'connecting' ? '◐ Connecting' : 
                   '○ Offline'}
                </span>
              </div>
            </div>

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

            {/* Bottom-left: Audio/Video controls - ALWAYS VISIBLE when connected */}
            {appState === "connected" && (
              <div className="absolute bottom-6 left-6 flex gap-3">
                <button
                  onClick={toggleMicrophone}
                  className={`
                    px-4 py-2 rounded-full transition-all duration-300 flex items-center gap-2
                    ${isDarkMode 
                      ? (isMicEnabled ? 'bg-white/90 text-black' : 'bg-white/50 text-black/50') 
                      : (isMicEnabled ? 'bg-black/90 text-white' : 'bg-black/50 text-white/50')}
                    hover:scale-[1.02] active:scale-[0.98] transform
                  `}
                >
                  <span className="text-sm">
                    {isMicEnabled ? '🎤' : '🔇'}
                  </span>
                  <span className="font-light text-xs tracking-[0.15em] uppercase">
                    {isMicEnabled ? 'Mic On' : 'Mic Off'}
                  </span>
                </button>
                
                <button
                  onClick={toggleCamera}
                  className={`
                    px-4 py-2 rounded-full transition-all duration-300 flex items-center gap-2
                    ${isDarkMode 
                      ? (isCameraEnabled ? 'bg-white/90 text-black' : 'bg-white/50 text-black/50') 
                      : (isCameraEnabled ? 'bg-black/90 text-white' : 'bg-black/50 text-white/50')}
                    hover:scale-[1.02] active:scale-[0.98] transform
                  `}
                >
                  <span className="text-sm">
                    {isCameraEnabled ? '📹' : '📷'}
                  </span>
                  <span className="font-light text-xs tracking-[0.15em] uppercase">
                    {isCameraEnabled ? 'Camera On' : 'Camera Off'}
                  </span>
                </button>
              </div>
            )}

            {/* Bottom-center: Disconnect button when connected */}
            {appState === "connected" && (
              <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2">
                <button
                  onClick={handleDisconnect}
                  className={`
                    px-4 py-2 rounded-full transition-all duration-300
                    ${isDarkMode 
                      ? 'bg-white/90 text-black' 
                      : 'bg-black/90 text-white'}
                    hover:scale-[1.02] active:scale-[0.98] transform
                  `}
                >
                  <span className="font-light text-xs tracking-[0.15em] uppercase">
                    Disconnect
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
          <VoiceReactivePlasma isDarkMode={isDarkMode} />
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
            <div className={`
              px-6 pb-3 overflow-y-auto
              ${transcriptExpanded ? 'max-h-[calc(100vh-12rem)]' : 'max-h-24'}
            `}>
              {/* Show conversation history if available */}
              {conversationHistory.length > 0 ? (
                <div className="space-y-3">
                  {conversationHistory.map((message, idx) => (
                    <div key={idx} className="flex gap-3 mb-3">
                      <span className={`
                        font-bold uppercase tracking-wider min-w-[3rem]
                        ${isDarkMode ? 'text-black' : 'text-white'}
                        ${fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-sm' : 'text-xs'}
                      `}>
                        {message.role === 'user' ? 'USER' : 'AI'}
                      </span>
                      <p className={`
                        flex-1
                        ${message.role === 'user'
                          ? (isDarkMode ? 'text-black/70' : 'text-white/70')
                          : (isDarkMode ? 'text-black/90' : 'text-white/90')
                        }
                        ${fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-base' : 'text-sm'}
                      `}>
                        {message.text}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <TranscriptOverlay
                  participant="remote"
                  className={`
                    ${isDarkMode ? 'text-white/80' : 'text-black/80'}
                    ${fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-base' : 'text-sm'}
                  `}
                />
              )}
            </div>
          </div>
        )}
      </FullScreenContainer>
      
      {/* Audio component */}
      <PipecatClientAudio />
    </PipecatClientProvider>
  );
}