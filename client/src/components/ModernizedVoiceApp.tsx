"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { VoiceApp } from './VoiceApp';
import { ModernVoiceChat } from './ModernVoiceChat';
import { EnhancedExport } from './EnhancedExport';
import { motion, AnimatePresence } from 'framer-motion';

interface ModernizedVoiceAppProps {
  videoEnabled: boolean;
}

type ViewMode = 'modern' | 'classic';
type ConversationMessage = {
  role: 'user' | 'assistant';
  text: string;
  id: string;
  timestamp?: Date;
};

export function ModernizedVoiceApp({ videoEnabled }: ModernizedVoiceAppProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('modern');
  const [showExportPanel, setShowExportPanel] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  
  // Bridge state between classic VoiceApp and modern components
  const [bridgeState, setBridgeState] = useState({
    conversationHistory: [] as ConversationMessage[],
    currentUserTranscript: '',
    currentAssistantTranscript: '',
    isConnected: false,
    isAssistantSpeaking: false,
    isUserSpeaking: false,
    isAssistantThinking: false,
  });

  // Export handler
  const handleExportRequest = useCallback(() => {
    setShowExportPanel(!showExportPanel);
  }, [showExportPanel]);

  // Mode toggle with smooth animation
  const toggleViewMode = useCallback(() => {
    setViewMode(current => current === 'modern' ? 'classic' : 'modern');
  }, []);

  // Dark mode toggle
  const toggleDarkMode = useCallback(() => {
    setIsDarkMode(current => !current);
  }, []);

  if (viewMode === 'classic') {
    return (
      <div className="relative w-full h-full">
        {/* Mode Toggle Button - Classic View */}
        <div className="absolute top-6 right-6 z-[60]">
          <motion.button
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            onClick={toggleViewMode}
            className={`px-4 py-2 rounded-full transition-all duration-300 backdrop-blur-sm border ${
              isDarkMode 
                ? 'bg-white/90 hover:bg-white text-black border-white/20' 
                : 'bg-black/90 hover:bg-black text-white border-black/20'
            }`}
          >
            <span className="font-light text-xs tracking-[0.15em] uppercase">
              🚀 Modern UI
            </span>
          </motion.button>
        </div>

        <VoiceApp videoEnabled={videoEnabled} />
      </div>
    );
  }

  return (
    <div className={`modernized-voice-app w-full h-full relative ${
      isDarkMode ? 'dark bg-gray-900' : 'bg-white'
    }`}>
      {/* Header Controls */}
      <div className="absolute top-0 left-0 right-0 z-50 flex items-center justify-between p-4 backdrop-blur-sm border-b border-opacity-20">
        <div className="flex items-center gap-4">
          <h1 className={`text-xl font-bold ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>
            🎤 Slowcat
          </h1>
          <div className={`text-sm ${
            isDarkMode ? 'text-gray-400' : 'text-gray-600'
          }`}>
            Voice Agent • Modern UI
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Dark Mode Toggle */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleDarkMode}
            className={`w-10 h-10 rounded-full transition-all duration-300 flex items-center justify-center ${
              isDarkMode 
                ? 'bg-yellow-500 text-black hover:bg-yellow-400' 
                : 'bg-gray-800 text-white hover:bg-gray-700'
            }`}
          >
            <span className="text-sm">{isDarkMode ? '☀️' : '🌙'}</span>
          </motion.button>
          
          {/* Export Toggle */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleExportRequest}
            className={`px-4 py-2 rounded-full transition-all duration-300 flex items-center gap-2 ${
              showExportPanel
                ? isDarkMode 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-blue-500 text-white'
                : isDarkMode
                  ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            <span className="text-sm">📥</span>
            <span className="font-light text-xs tracking-[0.15em] uppercase">
              Export
            </span>
          </motion.button>
          
          {/* View Mode Toggle */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleViewMode}
            className={`px-4 py-2 rounded-full transition-all duration-300 ${
              isDarkMode 
                ? 'bg-white text-black hover:bg-gray-100' 
                : 'bg-black text-white hover:bg-gray-900'
            }`}
          >
            <span className="font-light text-xs tracking-[0.15em] uppercase">
              Classic UI
            </span>
          </motion.button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex h-full pt-20">
        {/* Chat Interface - Takes full width when export panel is closed */}
        <motion.div
          initial={false}
          animate={{ 
            width: showExportPanel ? '60%' : '100%',
            marginRight: showExportPanel ? '2rem' : '0'
          }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
          className="relative"
        >
          <ModernVoiceChat
            conversationHistory={bridgeState.conversationHistory}
            currentUserTranscript={bridgeState.currentUserTranscript}
            currentAssistantTranscript={bridgeState.currentAssistantTranscript}
            isConnected={bridgeState.isConnected}
            isDarkMode={isDarkMode}
            isAssistantSpeaking={bridgeState.isAssistantSpeaking}
            isUserSpeaking={bridgeState.isUserSpeaking}
            isAssistantThinking={bridgeState.isAssistantThinking}
            onExportRequest={handleExportRequest}
            className="h-full"
          />
        </motion.div>

        {/* Export Panel - Slides in from right */}
        <AnimatePresence>
          {showExportPanel && (
            <motion.div
              initial={{ x: '100%', opacity: 0 }}
              animate={{ x: '0%', opacity: 1 }}
              exit={{ x: '100%', opacity: 0 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="absolute right-0 top-0 bottom-0 w-[38%] p-4 overflow-y-auto"
            >
              <EnhancedExport
                conversationHistory={bridgeState.conversationHistory}
                isDarkMode={isDarkMode}
                className="h-full"
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Demo Data Button - For Testing (Remove in Production) */}
      {process.env.NODE_ENV === 'development' && (
        <div className="absolute bottom-6 left-6 z-50">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              // Add demo conversation data for testing
              const demoMessages: ConversationMessage[] = [
                {
                  id: 'demo-1',
                  role: 'user',
                  text: 'Hello, can you help me with my project?',
                  timestamp: new Date(Date.now() - 300000) // 5 minutes ago
                },
                {
                  id: 'demo-2',
                  role: 'assistant', 
                  text: 'Of course! I\'d be happy to help you with your project. What kind of project are you working on, and what specific assistance do you need?',
                  timestamp: new Date(Date.now() - 280000)
                },
                {
                  id: 'demo-3',
                  role: 'user',
                  text: 'I\'m building a voice interface application and need help with the UI components.',
                  timestamp: new Date(Date.now() - 260000)
                },
                {
                  id: 'demo-4',
                  role: 'assistant',
                  text: 'That sounds like an exciting project! Voice interfaces are becoming increasingly important. For UI components, I\'d recommend using battle-tested libraries like Chatscope for chat interfaces and ensure you have real-time streaming text display.',
                  timestamp: new Date(Date.now() - 240000)
                }
              ];

              setBridgeState(prev => ({
                ...prev,
                conversationHistory: demoMessages,
                isConnected: true
              }));
            }}
            className={`px-3 py-2 rounded-lg text-xs ${
              isDarkMode
                ? 'bg-purple-600 text-white hover:bg-purple-700'
                : 'bg-purple-500 text-white hover:bg-purple-600'
            }`}
          >
Load Demo Data
          </button>
        </div>
      )}

      {/* Streaming Demo Button - For Testing TTS Sync */}
      {process.env.NODE_ENV === 'development' && (
        <div className="absolute bottom-20 left-6 z-50">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              // Simulate streaming assistant response
              setBridgeState(prev => ({ ...prev, isAssistantSpeaking: true }));
              
              const fullText = "This is a demonstration of streaming text that appears as the assistant is speaking. You can see each word appearing in real-time, synchronized with the text-to-speech synthesis.";
              let currentText = "";
              let index = 0;
              
              const streamInterval = setInterval(() => {
                if (index < fullText.length) {
                  currentText += fullText[index];
                  setBridgeState(prev => ({
                    ...prev,
                    currentAssistantTranscript: currentText,
                    isAssistantSpeaking: true
                  }));
                  index++;
                } else {
                  // Finish streaming
                  clearInterval(streamInterval);
                  setTimeout(() => {
                    // Add to conversation history
                    const newMessage: ConversationMessage = {
                      id: `demo-streaming-${Date.now()}`,
                      role: 'assistant',
                      text: fullText,
                      timestamp: new Date()
                    };
                    
                    setBridgeState(prev => ({
                      ...prev,
                      conversationHistory: [...prev.conversationHistory, newMessage],
                      currentAssistantTranscript: '',
                      isAssistantSpeaking: false
                    }));
                  }, 500);
                }
              }, 50); // 50ms per character = ~20 chars/second
            }}
            className={`px-3 py-2 rounded-lg text-xs ${
              isDarkMode
                ? 'bg-green-600 text-white hover:bg-green-700'
                : 'bg-green-500 text-white hover:bg-green-600'
            }`}
          >
            🎵 Demo TTS Stream
          </button>
        </div>
      )}
    </div>
  );
}

// Hook for integrating with existing VoiceApp state
export function useVoiceAppBridge() {
  const [bridgeState, setBridgeState] = useState({
    conversationHistory: [] as ConversationMessage[],
    currentUserTranscript: '',
    currentAssistantTranscript: '',
    isConnected: false,
    isAssistantSpeaking: false,
    isUserSpeaking: false,
    isAssistantThinking: false,
  });

  // Integration functions that can be called from VoiceApp
  const addToHistory = useCallback((message: ConversationMessage) => {
    setBridgeState(prev => ({
      ...prev,
      conversationHistory: [...prev.conversationHistory, message]
    }));
  }, []);

  const setUserTranscript = useCallback((transcript: string) => {
    setBridgeState(prev => ({ ...prev, currentUserTranscript: transcript }));
  }, []);

  const setAssistantTranscript = useCallback((transcript: string) => {
    setBridgeState(prev => ({ ...prev, currentAssistantTranscript: transcript }));
  }, []);

  const setConnectionState = useCallback((isConnected: boolean) => {
    setBridgeState(prev => ({ ...prev, isConnected }));
  }, []);

  const setVoiceState = useCallback((state: {
    isAssistantSpeaking?: boolean;
    isUserSpeaking?: boolean; 
    isAssistantThinking?: boolean;
  }) => {
    setBridgeState(prev => ({ ...prev, ...state }));
  }, []);

  return {
    bridgeState,
    addToHistory,
    setUserTranscript,
    setAssistantTranscript,
    setConnectionState,
    setVoiceState
  };
}