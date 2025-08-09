"use client";

import { useEffect, useRef, useState } from 'react';

interface StreamingTextProps {
  text: string;
  className?: string;
  speed?: number; // Characters per second
  isComplete?: boolean;
  onComplete?: () => void;
}

export function StreamingText({ 
  text, 
  className = '', 
  speed = 50, // 50 chars per second by default
  isComplete = false,
  onComplete 
}: StreamingTextProps) {
  const [displayedText, setDisplayedText] = useState('');
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const indexRef = useRef(0);
  const lastTextRef = useRef('');

  useEffect(() => {
    // Only run if text actually changed
    if (text === lastTextRef.current) {
      return;
    }

    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    if (!text) {
      setDisplayedText('');
      indexRef.current = 0;
      lastTextRef.current = text;
      return;
    }

    if (isComplete) {
      // If marked as complete, show all text immediately
      setDisplayedText(text);
      indexRef.current = text.length;
      lastTextRef.current = text;
      onComplete?.();
      return;
    }

    // Reset index if text is completely new or shorter
    if (text.length < lastTextRef.current.length || !text.startsWith(lastTextRef.current)) {
      indexRef.current = 0;
      setDisplayedText('');
    }

    // Start/continue animation from current index
    const startIndex = indexRef.current;
    
    const animateText = () => {
      if (indexRef.current < text.length) {
        setDisplayedText(text.slice(0, indexRef.current + 1));
        indexRef.current++;
        
        // Calculate delay based on speed (chars per second)
        const delay = 1000 / speed;
        timeoutRef.current = setTimeout(animateText, delay);
      } else {
        timeoutRef.current = null;
        if (isComplete) {
          onComplete?.();
        }
      }
    };

    // Only start animation if we have more text to show
    if (startIndex < text.length) {
      animateText();
    }

    lastTextRef.current = text;

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [text, speed, isComplete, onComplete]);

  return (
    <span className={className}>
      {displayedText}
      {!isComplete && displayedText.length > 0 && (
        <span className="animate-pulse opacity-70">|</span>
      )}
    </span>
  );
}

interface StreamingTextWordProps {
  text: string;
  className?: string;
  speed?: number; // Words per second
  isComplete?: boolean;
  onComplete?: () => void;
}

export function StreamingTextWord({ 
  text, 
  className = '', 
  speed = 8, // 8 words per second by default
  isComplete = false,
  onComplete 
}: StreamingTextWordProps) {
  const [displayedText, setDisplayedText] = useState('');
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const wordIndexRef = useRef(0);

  useEffect(() => {
    if (!text) {
      setDisplayedText('');
      wordIndexRef.current = 0;
      return;
    }

    if (isComplete) {
      // If marked as complete, show all text immediately
      setDisplayedText(text);
      onComplete?.();
      return;
    }

    const words = text.split(' ');
    
    // If we have more words to display and we're not currently animating
    if (words.length > wordIndexRef.current && !timeoutRef.current) {
      const animateText = () => {
        if (wordIndexRef.current < words.length) {
          const newText = words.slice(0, wordIndexRef.current + 1).join(' ');
          setDisplayedText(newText);
          wordIndexRef.current++;
          
          // Calculate delay based on speed (words per second)
          const delay = 1000 / speed;
          timeoutRef.current = setTimeout(animateText, delay);
        } else {
          timeoutRef.current = null;
          // Don't call onComplete for partial updates, only when truly complete
          if (isComplete) {
            onComplete?.();
          }
        }
      };

      animateText();
    }

    // If text becomes shorter (new sentence started), reset
    const displayedWords = displayedText.split(' ').length;
    if (words.length < displayedWords) {
      setDisplayedText('');
      wordIndexRef.current = 0;
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [text, speed, isComplete, onComplete, displayedText]);

  return (
    <span className={className}>
      {displayedText}
      {!isComplete && displayedText.length > 0 && (
        <span className="animate-pulse opacity-70 ml-1">|</span>
      )}
    </span>
  );
}