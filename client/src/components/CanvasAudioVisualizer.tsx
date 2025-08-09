"use client";

import { useState, useEffect, useRef, useCallback } from 'react';

interface CanvasAudioVisualizerProps {
  isDarkMode: boolean;
  audioLevel?: number; // 0-1 for voice reactivity
  voiceState?: 'idle' | 'listening' | 'speaking' | 'thinking';
  frequencyData?: Uint8Array; // Real frequency data from Web Audio API
}

export function CanvasAudioVisualizer({ 
  isDarkMode, 
  audioLevel = 0,
  voiceState = 'idle',
  frequencyData
}: CanvasAudioVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const [isVisible, setIsVisible] = useState(true);
  
  // Debug log
  useEffect(() => {
    console.log('🎨 CanvasAudioVisualizer active - ultra-lightweight & beautiful');
  }, []);

  // Pause when not visible
  useEffect(() => {
    if (!canvasRef.current) return;
    const observer = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold: 0.1 }
    );
    observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, []);

  // Colors based on theme and voice state
  const getColors = useCallback(() => {
    const base = isDarkMode ? {
      bg: '#000000',
      primary: '#ffffff',
      secondary: '#666666'
    } : {
      bg: '#ffffff', 
      primary: '#000000',
      secondary: '#999999'
    };

    switch (voiceState) {
      case 'listening':
        return { ...base, accent: '#00ff88', glow: 'rgba(0, 255, 136, 0.6)' };
      case 'speaking':
        return { ...base, accent: '#ff4757', glow: 'rgba(255, 71, 87, 0.6)' };
      case 'thinking':
        return { ...base, accent: '#3742fa', glow: 'rgba(55, 66, 250, 0.6)' };
      default:
        return { ...base, accent: '#00d4ff', glow: 'rgba(0, 212, 255, 0.6)' };
    }
  }, [isDarkMode, voiceState]);

  // Animation loop
  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !isVisible) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const colors = getColors();
    const { width, height } = canvas;
    
    // Clear canvas
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);
    
    // Create mesmerizing patterns
    const time = Date.now() * 0.001; // Time in seconds
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Much smaller, more subtle circle
    const baseRadius = Math.min(width, height) * 0.05; // 5% instead of 15%
    const reactiveRadius = baseRadius * (1 + audioLevel * 1.5); // Less reactive
    
    // Subtle outer rings - fewer and smaller
    for (let i = 0; i < 3; i++) {
      const ringRadius = reactiveRadius * (1 + i * 0.5);
      const opacity = (1 - i * 0.3) * audioLevel * 0.4; // Much more subtle
      
      ctx.beginPath();
      ctx.arc(centerX, centerY, ringRadius, 0, Math.PI * 2);
      ctx.strokeStyle = colors.glow.replace('0.6', `${opacity}`);
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    
    // Smaller flowing plasma-like waves
    const waveCount = 5; // Fewer blobs
    for (let i = 0; i < waveCount; i++) {
      const angle = (i / waveCount) * Math.PI * 2 + time * 0.5; // Slower
      const waveRadius = reactiveRadius * (2 + Math.sin(time * 1.5 + i) * 0.5);
      
      const x = centerX + Math.cos(angle) * waveRadius;
      const y = centerY + Math.sin(angle) * waveRadius;
      
      // Much smaller flowing blobs
      const blobRadius = 8 + audioLevel * 12; // Smaller base size
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, blobRadius);
      gradient.addColorStop(0, colors.glow.replace('0.6', '0.3')); // More subtle
      gradient.addColorStop(1, 'transparent');
      
      ctx.beginPath();
      ctx.arc(x, y, blobRadius, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();
    }
    
    // Much smaller central pulsing core
    const coreGradient = ctx.createRadialGradient(
      centerX, centerY, 0,
      centerX, centerY, reactiveRadius
    );
    coreGradient.addColorStop(0, colors.accent.replace('#', '#') + '80'); // Semi-transparent
    coreGradient.addColorStop(0.7, colors.glow.replace('0.6', '0.2')); // Much more subtle
    coreGradient.addColorStop(1, 'transparent');
    
    ctx.beginPath();
    ctx.arc(centerX, centerY, reactiveRadius, 0, Math.PI * 2);
    ctx.fillStyle = coreGradient;
    ctx.fill();
    
    // Small frequency bars at bottom (if real audio data available)
    if (frequencyData) {
      const barWidth = width / (frequencyData.length / 8); // Fewer bars
      const maxBarHeight = height * 0.15; // Much smaller bars
      
      for (let i = 0; i < frequencyData.length; i += 8) { // Sample every 8th
        const barHeight = (frequencyData[i] / 255) * maxBarHeight;
        const x = (i / 8) * barWidth;
        const y = height - barHeight - 10; // 10px from bottom
        
        const gradient = ctx.createLinearGradient(x, height - 10, x, y);
        gradient.addColorStop(0, colors.accent + '40'); // Semi-transparent
        gradient.addColorStop(1, colors.glow.replace('0.6', '0.4'));
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, barWidth * 0.8, barHeight); // Thinner bars
      }
    }
    
    // Schedule next frame
    animationRef.current = requestAnimationFrame(animate);
  }, [isVisible, audioLevel, frequencyData, getColors]);

  // Start/stop animation
  useEffect(() => {
    if (isVisible) {
      animationRef.current = requestAnimationFrame(animate);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isVisible, animate]);

  // Handle canvas resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const resizeCanvas = () => {
      const container = canvas.parentElement;
      if (!container) return;
      
      const rect = container.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      
      // Set CSS size to match container
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      }
    };
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    return () => window.removeEventListener('resize', resizeCanvas);
  }, []);

  return (
    <div className="absolute inset-0">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{
          background: isDarkMode ? '#000000' : '#ffffff',
          imageRendering: 'auto',
          display: 'block'
        }}
      />
      
      {/* Debug info */}
      <div 
        className="absolute top-4 right-4 px-2 py-1 rounded text-xs font-mono opacity-50"
        style={{ color: isDarkMode ? '#ffffff' : '#000000' }}
      >
        Canvas | {voiceState} | {Math.round(audioLevel * 100)}%
      </div>
    </div>
  );
}