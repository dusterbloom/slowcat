"use client";

import React, { useEffect, useState, useRef } from 'react';
import { motion } from 'framer-motion';

interface StreamingMessageProps {
  text: string;
  isStreaming?: boolean;
  speed?: number; // Characters per second
  wordHighlight?: boolean; // Enable karaoke-style highlighting
  onWordSpoken?: (wordIndex: number) => void;
  className?: string;
}

export function StreamingMessage({ 
  text, 
  isStreaming = false, 
  speed = 30,
  wordHighlight = false,
  onWordSpoken,
  className = ""
}: StreamingMessageProps) {
  const [displayedText, setDisplayedText] = useState('');
  const [currentWordIndex, setCurrentWordIndex] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout>();
  const words = text.split(' ');

  useEffect(() => {
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    if (!isStreaming || !text) {
      setDisplayedText(text);
      return;
    }

    // Reset state for new streaming session
    setDisplayedText('');
    setCurrentWordIndex(0);
    
    let charIndex = 0;
    const displayChar = () => {
      if (charIndex < text.length) {
        const newDisplayText = text.slice(0, charIndex + 1);
        setDisplayedText(newDisplayText);
        
        // Track word progress for highlighting
        if (wordHighlight) {
          const currentWord = newDisplayText.split(' ').length - 1;
          if (currentWord !== currentWordIndex && currentWord < words.length) {
            setCurrentWordIndex(currentWord);
            onWordSpoken?.(currentWord);
          }
        }
        
        charIndex++;
      } else {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      }
    };

    intervalRef.current = setInterval(displayChar, 1000 / speed);
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [text, isStreaming, speed, wordHighlight, onWordSpoken]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  if (!wordHighlight) {
    return (
      <div className={`streaming-message ${className}`}>
        {displayedText}
        {isStreaming && displayedText.length < text.length && (
          <motion.span 
            animate={{ opacity: [0, 1] }}
            transition={{ repeat: Infinity, duration: 0.8 }}
            className="inline-block ml-1 w-2 h-4 bg-current"
            style={{ backgroundColor: 'currentColor' }}
          >
            |
          </motion.span>
        )}
      </div>
    );
  }

  return (
    <div className={`streaming-message-karaoke ${className}`}>
      {words.map((word, index) => (
        <motion.span
          key={`${word}-${index}`}
          className={`inline-block mr-1 ${index <= currentWordIndex ? 'spoken' : 'unspoken'}`}
          animate={{
            color: index <= currentWordIndex ? '#3B82F6' : '#64748B',
            scale: index === currentWordIndex ? 1.05 : 1
          }}
          transition={{ duration: 0.2 }}
          style={{
            textShadow: index === currentWordIndex ? '0 0 8px rgba(59, 130, 246, 0.5)' : 'none'
          }}
        >
          {word}
        </motion.span>
      ))}
      {isStreaming && (
        <motion.span 
          animate={{ opacity: [0, 1] }}
          transition={{ repeat: Infinity, duration: 0.8 }}
          className="inline-block ml-1 w-2 h-4 bg-blue-500"
        >
          |
        </motion.span>
      )}
    </div>
  );
}

// Voice indicator for showing current voice state
interface VoiceIndicatorProps {
  type: 'speaking' | 'listening' | 'synthesizing' | 'thinking';
  className?: string;
}

export function VoiceIndicator({ type, className = "" }: VoiceIndicatorProps) {
  const getIndicatorConfig = () => {
    switch (type) {
      case 'speaking':
        return { 
          emoji: '🎤', 
          text: 'Speaking...', 
          color: 'text-blue-500',
          pulseColor: 'bg-blue-500' 
        };
      case 'listening':
        return { 
          emoji: '👂', 
          text: 'Listening...', 
          color: 'text-green-500',
          pulseColor: 'bg-green-500' 
        };
      case 'synthesizing':
        return { 
          emoji: '🎵', 
          text: 'Speaking...', 
          color: 'text-purple-500',
          pulseColor: 'bg-purple-500' 
        };
      case 'thinking':
        return { 
          emoji: '💭', 
          text: 'Thinking...', 
          color: 'text-orange-500',
          pulseColor: 'bg-orange-500' 
        };
    }
  };

  const config = getIndicatorConfig();

  return (
    <div className={`voice-indicator flex items-center gap-2 text-xs mt-1 ${config.color} ${className}`}>
      <span className="text-sm">{config.emoji}</span>
      <span>{config.text}</span>
      <div className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className={`w-1 h-1 rounded-full ${config.pulseColor}`}
            animate={{ 
              scale: [1, 1.5, 1],
              opacity: [0.4, 1, 0.4] 
            }}
            transition={{
              duration: 1.2,
              repeat: Infinity,
              delay: i * 0.15
            }}
          />
        ))}
      </div>
    </div>
  );
}