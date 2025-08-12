"use client";

import React, { useCallback, useEffect, useState, useRef } from 'react';
import { 
  ChatContainer, 
  MessageList, 
  Message, 
  MessageInput,
  TypingIndicator,
  ConversationHeader,
  Avatar
} from '@chatscope/chat-ui-kit-react';
import { StreamingMessage, VoiceIndicator } from './StreamingMessage';

interface ModernVoiceChatProps {
  conversationHistory: Array<{
    role: 'user' | 'assistant';
    text: string;
    id: string;
    timestamp?: Date;
  }>;
  currentUserTranscript: string;
  currentAssistantTranscript: string;
  isConnected: boolean;
  isDarkMode: boolean;
  isAssistantSpeaking?: boolean;
  isUserSpeaking?: boolean;
  isAssistantThinking?: boolean;
  onExportRequest?: () => void;
  className?: string;
}

export function ModernVoiceChat({
  conversationHistory = [],
  currentUserTranscript,
  currentAssistantTranscript,
  isConnected,
  isDarkMode,
  isAssistantSpeaking = false,
  isUserSpeaking = false,
  isAssistantThinking = false,
  onExportRequest,
  className = ""
}: ModernVoiceChatProps) {
  const messageListRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (autoScroll && messageListRef.current) {
      const scrollElement = messageListRef.current;
      scrollElement.scrollTop = scrollElement.scrollHeight;
    }
  }, [conversationHistory, currentUserTranscript, currentAssistantTranscript, autoScroll]);

  // Detect if user has scrolled up to disable auto-scroll
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
    const isAtBottom = scrollHeight - scrollTop <= clientHeight + 50; // 50px tolerance
    setAutoScroll(isAtBottom);
  }, []);

  const getConnectionStatus = () => {
    if (!isConnected) return { text: "Disconnected", color: "text-red-500", emoji: "🔴" };
    if (isAssistantThinking) return { text: "Thinking", color: "text-orange-500", emoji: "💭" };
    if (isAssistantSpeaking) return { text: "Speaking", color: "text-purple-500", emoji: "🎵" };
    if (isUserSpeaking) return { text: "Listening", color: "text-green-500", emoji: "👂" };
    return { text: "Connected", color: "text-blue-500", emoji: "🟢" };
  };

  const status = getConnectionStatus();

  return (
    <div className={`modern-voice-chat h-full ${isDarkMode ? 'dark' : ''} ${className}`}>
      <ChatContainer className="h-full">
        <ConversationHeader>
          <ConversationHeader.Back />
          <Avatar 
            src="data:image/svg+xml,%3csvg width='100' height='100' xmlns='http://www.w3.org/2000/svg'%3e%3cdefs%3e%3clinearGradient id='a' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3e%3cstop offset='0%25' style='stop-color:%234F46E5'/%3e%3cstop offset='100%25' style='stop-color:%237C3AED'/%3e%3c/linearGradient%3e%3c/defs%3e%3ccircle cx='50' cy='50' r='50' fill='url(%23a)'/%3e%3ctext x='50%25' y='50%25' text-anchor='middle' dy='0.35em' fill='white' font-size='40' font-weight='bold'%3e🎤%3c/text%3e%3c/svg%3e"
            name="Slowcat"
          />
          <ConversationHeader.Content 
            userName="Slowcat Voice Agent"
            info={
              <div className={`flex items-center gap-2 text-xs ${status.color}`}>
                <span>{status.emoji}</span>
                <span>{status.text}</span>
              </div>
            }
          />
          <ConversationHeader.Actions>
            {onExportRequest && (
              <button
                onClick={onExportRequest}
                className="text-xs px-3 py-1 rounded bg-blue-500 text-white hover:bg-blue-600 transition-colors"
                title="Export conversation"
              >
                📥 Export
              </button>
            )}
          </ConversationHeader.Actions>
        </ConversationHeader>

        <MessageList
          ref={messageListRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto"
          autoScrollToBottom={autoScroll}
          autoScrollToBottomOnMount={true}
        >
          {/* Conversation History */}
          {conversationHistory.map((msg, index) => (
            <Message
              key={msg.id || `msg-${index}`}
              model={{
                message: msg.text,
                sentTime: msg.timestamp?.toISOString() || new Date().toISOString(),
                sender: msg.role === 'user' ? 'You' : 'Slowcat',
                direction: msg.role === 'user' ? 'outgoing' : 'incoming',
                position: 'single'
              }}
              className={msg.role === 'user' ? 'user-message' : 'assistant-message'}
            >
              <Message.CustomContent>
                <div className={`message-content ${
                  msg.role === 'user' 
                    ? 'bg-blue-500 text-white p-3 rounded-lg max-w-xs ml-auto' 
                    : isDarkMode 
                      ? 'bg-gray-700 text-gray-100 p-3 rounded-lg max-w-xs' 
                      : 'bg-gray-100 text-gray-900 p-3 rounded-lg max-w-xs'
                }`}>
                  <div className="text-sm leading-relaxed">
                    {msg.text}
                  </div>
                  {msg.timestamp && (
                    <div className={`text-xs mt-1 opacity-70 ${
                      msg.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                    }`}>
                      {msg.timestamp.toLocaleTimeString([], { 
                        hour: '2-digit', 
                        minute: '2-digit' 
                      })}
                    </div>
                  )}
                </div>
              </Message.CustomContent>
            </Message>
          ))}

          {/* Real-time User Speech (Streaming) */}
          {currentUserTranscript && (
            <Message
              model={{
                message: '',
                sender: 'You',
                direction: 'outgoing',
                position: 'single'
              }}
              className="streaming-user-message"
            >
              <Message.CustomContent>
                <div className="bg-blue-500 text-white p-3 rounded-lg max-w-xs ml-auto opacity-90">
                  <div className="text-sm leading-relaxed">
                    <StreamingMessage 
                      text={currentUserTranscript}
                      isStreaming={true}
                      speed={40} // Faster for real-time speech
                      className="text-white"
                    />
                  </div>
                  <VoiceIndicator 
                    type="speaking" 
                    className="text-blue-100 mt-1" 
                  />
                </div>
              </Message.CustomContent>
            </Message>
          )}

          {/* Real-time Assistant TTS (Streaming) - THE MAGIC! */}
          {currentAssistantTranscript && (
            <Message
              model={{
                message: '',
                sender: 'Slowcat',
                direction: 'incoming', 
                position: 'single'
              }}
              className="streaming-assistant-message"
            >
              <Message.CustomContent>
                <div className={`p-3 rounded-lg max-w-xs opacity-90 ${
                  isDarkMode 
                    ? 'bg-gray-700 text-gray-100' 
                    : 'bg-gray-100 text-gray-900'
                }`}>
                  <div className="text-sm leading-relaxed">
                    <StreamingMessage 
                      text={currentAssistantTranscript}
                      isStreaming={true}
                      speed={30} // Sync with TTS speed
                      wordHighlight={true} // Enable karaoke effect
                      className={isDarkMode ? 'text-gray-100' : 'text-gray-900'}
                    />
                  </div>
                  <VoiceIndicator 
                    type="synthesizing" 
                    className={`mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}
                  />
                </div>
              </Message.CustomContent>
            </Message>
          )}

          {/* Thinking indicator */}
          {isAssistantThinking && !currentAssistantTranscript && (
            <Message
              model={{
                message: '',
                sender: 'Slowcat',
                direction: 'incoming',
                position: 'single'
              }}
            >
              <Message.CustomContent>
                <div className={`p-3 rounded-lg max-w-xs ${
                  isDarkMode 
                    ? 'bg-gray-700 text-gray-100' 
                    : 'bg-gray-100 text-gray-900'
                }`}>
                  <VoiceIndicator 
                    type="thinking" 
                    className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}
                  />
                </div>
              </Message.CustomContent>
            </Message>
          )}

          {/* Empty state */}
          {conversationHistory.length === 0 && !currentUserTranscript && !currentAssistantTranscript && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center p-8">
                <div className="text-6xl mb-4">🎤</div>
                <h3 className={`text-lg font-semibold mb-2 ${
                  isDarkMode ? 'text-gray-100' : 'text-gray-900'
                }`}>
                  Ready to chat!
                </h3>
                <p className={`text-sm ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  {isConnected ? 
                    'Start speaking to begin your conversation' : 
                    'Connect to start your voice conversation'
                  }
                </p>
              </div>
            </div>
          )}
        </MessageList>

        {/* Voice-only - no text input needed, but show connection status */}
        <div className={`border-t p-3 text-center text-xs ${
          isDarkMode 
            ? 'border-gray-700 bg-gray-800 text-gray-400' 
            : 'border-gray-200 bg-gray-50 text-gray-600'
        }`}>
          <div className={`flex items-center justify-center gap-2 ${status.color}`}>
            <span>{status.emoji}</span>
            <span>
              {isConnected ? 
                'Voice chat active - speak naturally' : 
                'Voice chat disconnected'
              }
            </span>
          </div>
          {conversationHistory.length > 0 && (
            <div className="text-xs mt-1 opacity-70">
              {conversationHistory.length} messages in conversation
            </div>
          )}
        </div>
      </ChatContainer>

      <style jsx>{`
        .modern-voice-chat .cs-message-list {
          padding: 1rem;
        }
        
        .modern-voice-chat .cs-message {
          margin-bottom: 1rem;
        }
        
        .modern-voice-chat .cs-message__content {
          background: transparent !important;
          padding: 0 !important;
        }
        
        .modern-voice-chat.dark .cs-conversation-header {
          background-color: #1f2937;
          border-bottom-color: #374151;
          color: #f3f4f6;
        }
        
        .modern-voice-chat.dark .cs-message-list {
          background-color: #111827;
        }
        
        .modern-voice-chat.dark .cs-message-input {
          background-color: #1f2937;
          border-top-color: #374151;
        }
        
        /* Smooth scrolling */
        .cs-message-list {
          scroll-behavior: smooth;
        }
        
        /* Custom scrollbar for dark mode */
        .modern-voice-chat.dark .cs-message-list::-webkit-scrollbar {
          width: 6px;
        }
        
        .modern-voice-chat.dark .cs-message-list::-webkit-scrollbar-track {
          background: #1f2937;
        }
        
        .modern-voice-chat.dark .cs-message-list::-webkit-scrollbar-thumb {
          background: #4b5563;
          border-radius: 3px;
        }
        
        .modern-voice-chat.dark .cs-message-list::-webkit-scrollbar-thumb:hover {
          background: #6b7280;
        }
      `}</style>
    </div>
  );
}