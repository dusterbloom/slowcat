"use client";

import { useState, useEffect, useRef, useMemo } from 'react';
import { CanvasAudioVisualizer } from './CanvasAudioVisualizer';

interface UltraLightPlasmaProps {
  isDarkMode: boolean;
  voiceState?: 'idle' | 'listening' | 'speaking' | 'thinking';
  audioLevel?: number; // 0-1 for voice reactivity
  mode?: 'plasma' | 'canvas'; // Choose visualization mode
}

export function UltraLightPlasma({ 
  isDarkMode, 
  voiceState = 'idle', 
  audioLevel = 0,
  mode = 'canvas' // Default to new canvas visualizer
}: UltraLightPlasmaProps) {
  const [isVisible, setIsVisible] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);

  // Pause animation when not visible to save CPU
  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold: 0.1 }
    );

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Pause on window blur/focus to save battery
  useEffect(() => {
    const handleVisibilityChange = () => {
      setIsVisible(!document.hidden);
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, []);

  // Dynamic colors based on voice state and theme
  const getColors = () => {
    const base = isDarkMode ? {
      bg: '#000000',
      primary: '#FF6B35',
      secondary: '#8A2BE2',
      accent: '#00CED1',
      glow: 'rgba(255, 107, 53, 0.3)'
    } : {
      bg: '#FFFFFF', 
      primary: '#FF8C42',
      secondary: '#6A5ACD',
      accent: '#20B2AA',
      glow: 'rgba(255, 140, 66, 0.2)'
    };

    switch (voiceState) {
      case 'listening':
        return { ...base, primary: '#00FF41', glow: 'rgba(0, 255, 65, 0.4)' };
      case 'speaking':
        return { ...base, primary: '#FF4757', glow: 'rgba(255, 71, 87, 0.4)' };
      case 'thinking':
        return { ...base, primary: '#3742FA', glow: 'rgba(55, 66, 250, 0.4)' };
      default:
        return base;
    }
  };

  const colors = getColors();

  // Voice-reactive scaling
  const voiceScale = 1 + (audioLevel * 0.3);
  const pulseIntensity = 0.5 + (audioLevel * 0.5);

  // Memoized particles to prevent re-creation
  const particles = useMemo(() => {
    return Array.from({ length: 8 }, (_, i) => ({
      id: i,
      size: 20 + (i % 3) * 15,
      delay: i * 0.3,
      duration: 3 + (i % 2) * 2,
      x: 10 + (i * 12) % 80,
      y: 15 + (i * 17) % 70,
    }));
  }, []);

  if (!isVisible) {
    return <div className="absolute inset-0" style={{ backgroundColor: colors.bg }} />;
  }

  // Show canvas audio visualizer mode
  if (mode === 'canvas') {
    return (
      <CanvasAudioVisualizer
        isDarkMode={isDarkMode}
        voiceState={voiceState}
        audioLevel={audioLevel}
      />
    );
  }

  // Original plasma mode
  return (
    <div 
      ref={containerRef}
      className="absolute inset-0 overflow-hidden"
      style={{ backgroundColor: colors.bg }}
    >
      {/* Main plasma background - pure CSS gradients */}
      <div 
        className="absolute inset-0"
        style={{
          background: `
            radial-gradient(circle at 30% 20%, ${colors.primary}22 0%, transparent 50%),
            radial-gradient(circle at 80% 40%, ${colors.secondary}18 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, ${colors.accent}15 0%, transparent 50%),
            radial-gradient(circle at 90% 90%, ${colors.primary}12 0%, transparent 40%)
          `,
          animation: isVisible ? 'plasma-drift 20s ease-in-out infinite' : 'none',
          transform: `scale(${voiceScale})`,
          transition: 'transform 0.3s ease-out'
        }}
      />

      {/* Floating particles - hardware accelerated */}
      {particles.map(particle => (
        <div
          key={particle.id}
          className="absolute rounded-full"
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            width: particle.size,
            height: particle.size,
            background: `radial-gradient(circle, ${colors.primary}40 0%, transparent 70%)`,
            animation: isVisible ? `float-${particle.id} ${particle.duration}s ease-in-out infinite` : 'none',
            animationDelay: `${particle.delay}s`,
            transform: 'translateZ(0)', // Force GPU acceleration
            opacity: pulseIntensity,
            transition: 'opacity 0.2s ease-out'
          }}
        />
      ))}

      {/* Central pulse - voice reactive */}
      <div 
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{
          width: 120 + (audioLevel * 80),
          height: 120 + (audioLevel * 80),
          background: `radial-gradient(circle, ${colors.glow} 0%, transparent 70%)`,
          animation: isVisible ? 'central-pulse 2s ease-in-out infinite' : 'none',
          transform: 'translate(-50%, -50%) translateZ(0)',
          transition: 'width 0.1s ease-out, height 0.1s ease-out'
        }}
      />

      {/* Voice state indicator */}
      {voiceState !== 'idle' && (
        <div 
          className="absolute top-4 left-4 px-3 py-1 rounded-full text-sm font-medium"
          style={{
            backgroundColor: colors.primary + '40',
            color: colors.primary,
            backdropFilter: 'blur(10px)',
            border: `1px solid ${colors.primary}60`
          }}
        >
          {voiceState === 'listening' ? '🎤 Listening' : 
           voiceState === 'speaking' ? '🔊 Speaking' : 
           '💭 Thinking'}
        </div>
      )}

      {/* CSS Keyframes injected via style tag */}
      <style jsx>{`
        @keyframes plasma-drift {
          0%, 100% { 
            transform: scale(${voiceScale}) rotate(0deg) translate(0px, 0px); 
          }
          25% { 
            transform: scale(${voiceScale * 1.05}) rotate(1deg) translate(5px, -3px); 
          }
          50% { 
            transform: scale(${voiceScale * 0.95}) rotate(-1deg) translate(-3px, 5px); 
          }
          75% { 
            transform: scale(${voiceScale * 1.02}) rotate(0.5deg) translate(2px, -7px); 
          }
        }
        
        @keyframes central-pulse {
          0%, 100% { 
            opacity: ${pulseIntensity * 0.6}; 
            transform: translate(-50%, -50%) scale(1) translateZ(0); 
          }
          50% { 
            opacity: ${pulseIntensity}; 
            transform: translate(-50%, -50%) scale(1.1) translateZ(0); 
          }
        }
        
        ${particles.map(particle => `
          @keyframes float-${particle.id} {
            0%, 100% { 
              transform: translateZ(0) translate(0px, 0px) scale(1); 
              opacity: ${pulseIntensity * 0.4}; 
            }
            50% { 
              transform: translateZ(0) translate(${(particle.id % 2) * 10 - 5}px, ${(particle.id % 3) * 8 - 4}px) scale(1.2); 
              opacity: ${pulseIntensity * 0.8}; 
            }
          }
        `).join('')}
      `}</style>
    </div>
  );
}