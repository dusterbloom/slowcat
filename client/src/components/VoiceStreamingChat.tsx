"use client";

import React from 'react';
import { useChat } from 'ai/react';
import { StreamingMessage } from './StreamingMessage';

interface VoiceStreamingChatProps {
  onTtsText?: (text: string) => void;
  isDarkMode?: boolean;
  fontSize?: 'small' | 'normal' | 'large';
}

export function VoiceStreamingChat({ 
  onTtsText, 
  isDarkMode = false, 
  fontSize = 'normal' 
}: VoiceStreamingChatProps) {
  const { messages, isLoading } = useChat({
    api: '/api/chat',
    onFinish: (message) => {
      // Send final message to TTS
      onTtsText?.(message.content);
    }
  });

  return (
    <div className="voice-streaming-chat">
      {messages.map((message, idx) => (
        <div key={message.id || idx} className={`flex ${
          message.role === 'user' ? 'justify-end' : 'justify-start'
        } mb-3`}>
          <div className={`max-w-[80%] rounded-lg p-3 ${
            message.role === 'user' 
              ? 'bg-black text-white' 
              : (isDarkMode ? 'bg-gray-700 text-gray-100 border border-gray-600' : 'bg-gray-200 text-gray-900 border border-gray-300')
          }`}>
            <div className={`leading-relaxed break-words ${
              fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-base' : 'text-sm'
            }`}>
              {message.role === 'assistant' && isLoading && idx === messages.length - 1 ? (
                <StreamingMessage 
                  text={message.content}
                  isStreaming={true}
                  speed={30}
                  wordHighlight={true}
                  className={isDarkMode ? 'text-gray-100' : 'text-gray-900'}
                />
              ) : (
                message.content
              )}
            </div>
            {message.createdAt && (
              <div className={`text-xs mt-1 opacity-70 ${
                message.role === 'user' ? 'text-white' : (isDarkMode ? 'text-gray-400' : 'text-gray-500')
              }`}>
                {new Date(message.createdAt).toLocaleTimeString([], { 
                  hour: '2-digit', 
                  minute: '2-digit' 
                })}
              </div>
            )}
          </div>
        </div>
      ))}
      
      {isLoading && messages.length === 0 && (
        <div className="flex justify-start mb-3">
          <div className={`max-w-[80%] rounded-lg p-3 ${
            isDarkMode ? 'bg-gray-700 text-gray-100 border border-gray-600' : 'bg-gray-200 text-gray-900 border border-gray-300'
          } opacity-90`}>
            <div className={`leading-relaxed break-words ${
              fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-base' : 'text-sm'
            }`}>
              <StreamingMessage 
                text="..."
                isStreaming={true}
                speed={30}
                className={isDarkMode ? 'text-gray-100' : 'text-gray-900'}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}