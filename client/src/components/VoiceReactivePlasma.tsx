"use client";

import { useState, useEffect, useRef, useMemo } from 'react';
import FallbackPlasma from './FallbackPlasma';

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
  isLowPowerMode: boolean;
}

export function VoiceReactivePlasma({ isDarkMode, isLowPowerMode }: VoiceReactivePlasmaProps) {
  const [isVisible, setIsVisible] = useState(true);
  const [useReducedMotion, setUseReducedMotion] = useState(false);
  const [forceFallback, setForceFallback] = useState(false);
  const [cpuOverloaded, setCpuOverloaded] = useState(false);
  const [qualityLevel, setQualityLevel] = useState<'high' | 'medium' | 'low'>('medium'); // Start with medium quality

  // Read env var to force disable WebGL
  useEffect(() => {
    if (process.env.NEXT_PUBLIC_DISABLE_WEBGL_PLASMA === 'true') {
      setForceFallback(true);
    }
  }, []);

  // Note: Low power mode is now controlled by parent component

  // Enhanced FPS monitor with adaptive quality
  useEffect(() => {
    if (isLowPowerMode) return; // Skip monitoring in low power mode
    
    let lastTime = performance.now();
    let frames = 0;
    let slowFrames = 0;
    let goodFrames = 0;
    let monitoring = true;

    function monitor(now: number) {
      frames++;
      const delta = now - lastTime;
      if (delta >= 1000) {
        const fps = (frames * 1000) / delta;
        frames = 0;
        lastTime = now;
        
        // Adaptive quality based on FPS
        if (fps < 15) {
          slowFrames++;
          goodFrames = 0;
          // Downgrade quality after 2 seconds of poor performance
          if (slowFrames >= 2) {
            if (qualityLevel === 'high') {
              setQualityLevel('medium');
              console.log('📉 Reducing WebGL quality to medium (FPS:', fps, ')');
            } else if (qualityLevel === 'medium') {
              setQualityLevel('low');
              console.log('📉 Reducing WebGL quality to low (FPS:', fps, ')');
            } else if (slowFrames >= 5) {
              // If still struggling after 5 seconds on low, fallback
              setCpuOverloaded(true);
              console.log('🔌 Switching to fallback renderer (FPS:', fps, ')');
            }
            slowFrames = 0;
          }
        } else if (fps > 30) {
          goodFrames++;
          slowFrames = 0;
          // Upgrade quality after 5 seconds of good performance
          if (goodFrames >= 5) {
            if (qualityLevel === 'low') {
              setQualityLevel('medium');
              console.log('📈 Increasing WebGL quality to medium (FPS:', fps, ')');
            } else if (qualityLevel === 'medium' && fps > 50) {
              setQualityLevel('high');
              console.log('📈 Increasing WebGL quality to high (FPS:', fps, ')');
            }
            goodFrames = 0;
          }
        } else {
          slowFrames = 0;
          goodFrames = 0;
        }
      }
      if (monitoring) requestAnimationFrame(monitor);
    }
    requestAnimationFrame(monitor);
    return () => {
      monitoring = false;
    };
  }, [qualityLevel, isLowPowerMode]);
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

  // Decide if fallback should be shown
  const shouldShowFallback = !PlasmaVisualizer || useReducedMotion || forceFallback || cpuOverloaded || isLowPowerMode;
  // === Throttled render loop control for WebGL plasma ===
  const lastRenderTimeRef = useRef<number>(0);
  // Adaptive frame rate based on quality level
  const TARGET_FPS: number = qualityLevel === 'high' ? 30 : qualityLevel === 'medium' ? 20 : 10;
  const FRAME_INTERVAL: number = 1000 / TARGET_FPS;
  useEffect(() => {
    if (!PlasmaVisualizer || shouldShowFallback) return;
    let rafId: number;
    const renderLoop = (time: number) => {
      // Only proceed if enough time has passed since last render
      if (typeof time === 'number' && time - lastRenderTimeRef.current >= FRAME_INTERVAL) {
        lastRenderTimeRef.current = time;
        // PlasmaVisualizer manages its own draw; throttling here prevents excessive calls
      }
      rafId = window.requestAnimationFrame(renderLoop);
    };
    rafId = window.requestAnimationFrame(renderLoop);
    return () => window.cancelAnimationFrame(rafId);
  }, [PlasmaVisualizer, shouldShowFallback]);
  // CPU overload detection (parent component can use this for automatic low power mode)
  // For now, we just detect but don't automatically switch modes

  if (shouldShowFallback) {
    return (
      <div className="absolute inset-0">
        <FallbackPlasma isDarkMode={isDarkMode} />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`absolute inset-0 ${isDarkMode ? 'bg-black plasma-dark' : 'bg-white plasma-light'}`}
    >
      {isVisible ? (
        PlasmaVisualizer ? (
          <div style={{
            width: '100%',
            height: '100%',
            transform: qualityLevel === 'low' ? 'scale(1.5)' : qualityLevel === 'medium' ? 'scale(1.2)' : 'none',
            transformOrigin: 'center',
          }}>
            <PlasmaVisualizer
              state="connected"
              style={{
                backgroundColor: isDarkMode ? 'black' : 'white',
                width: qualityLevel === 'low' ? '66%' : qualityLevel === 'medium' ? '83%' : '100%',
                height: qualityLevel === 'low' ? '66%' : qualityLevel === 'medium' ? '83%' : '100%',
                // Performance optimizations
                imageRendering: qualityLevel === 'low' ? 'pixelated' : 'auto',
                filter: qualityLevel === 'low' ? 'blur(1px)' : qualityLevel === 'medium' ? 'blur(0.5px)' : 'none',
                willChange: 'transform',
                transform: 'translateZ(0)', // Force GPU acceleration
                backfaceVisibility: 'hidden',
              }}
            />
          </div>
        ) : (
          <FallbackPlasma isDarkMode={isDarkMode} />
        )
      ) : (
        <FallbackPlasma isDarkMode={isDarkMode} />
      )}
    </div>
  );
}