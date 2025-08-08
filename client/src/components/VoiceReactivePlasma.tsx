"use client";

import { useState, useEffect, useRef, useMemo } from 'react';

// Import PlasmaVisualizer
let PlasmaVisualizer: React.ComponentType<{ state: string; style?: React.CSSProperties }> | null = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const webglModule = require("@pipecat-ai/voice-ui-kit/webgl");
  PlasmaVisualizer = webglModule.PlasmaVisualizer;
} catch {
  console.log("WebGL visualizer not available");
}

interface VoiceReactivePlasmaProps {
  isDarkMode: boolean;
}

export function VoiceReactivePlasma({ isDarkMode }: VoiceReactivePlasmaProps) {
  const [isVisible, setIsVisible] = useState(true);
  const [useReducedMotion, setUseReducedMotion] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const intersectionRef = useRef<IntersectionObserver | null>(null);

  // Check for user's motion preference
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setUseReducedMotion(mediaQuery.matches);
    
    const handler = (e: MediaQueryListEvent) => setUseReducedMotion(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  // Pause animation when not visible to save CPU
  useEffect(() => {
    if (!containerRef.current) return;

    intersectionRef.current = new IntersectionObserver(
      ([entry]) => {
        setIsVisible(entry.isIntersecting);
      },
      { threshold: 0.1 }
    );

    intersectionRef.current.observe(containerRef.current);
    return () => intersectionRef.current?.disconnect();
  }, []);

  // Pause on window blur/focus to save battery
  useEffect(() => {
    const handleVisibilityChange = () => {
      setIsVisible(!document.hidden);
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, []);

  // Memoized fallback gradient
  const fallbackGradient = useMemo(() => (
    <div className={`absolute inset-0 animate-pulse ${
      isDarkMode 
        ? 'bg-gradient-to-br from-orange-900/20 to-purple-900/20' 
        : 'bg-gradient-to-br from-orange-200/50 to-yellow-200/50'
    }`} />
  ), [isDarkMode]);

  if (!PlasmaVisualizer || useReducedMotion) {
    return fallbackGradient;
  }

  return (
    <div 
      ref={containerRef}
      className={`absolute inset-0 ${isDarkMode ? 'bg-black plasma-dark' : 'bg-white plasma-light'}`}
    >
      {isVisible ? (
        <PlasmaVisualizer 
          state="connected"
          style={{
            backgroundColor: isDarkMode ? 'black' : 'white',
            // Reduce quality slightly for better performance
            filter: 'blur(0.5px)',
          }}
        />
      ) : (
        fallbackGradient
      )}
    </div>
  );
}