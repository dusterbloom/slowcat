"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { usePipecatClientMediaTrack } from '@pipecat-ai/client-react';
import { usePipecatClient, VoiceVisualizer } from '@pipecat-ai/client-react';

type PermissionState = 'idle' | 'pending' | 'granted' | 'denied';

interface Blob {
  id: number;
  size: number;
  color: string;
}

interface BlobPhysics {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

interface VoiceReactiveVisualizerProps {
  isDarkMode: boolean;
  autoStart?: boolean;
  audioSource?: 'microphone' | 'output' | 'both';
  skipMicrophoneRequest?: boolean;
  externalShowControls?: boolean;
}

const NUM_BLOBS = 5;
const MIN_SPEED = 0.5;
const MAX_SPEED = 1.5;
const MIN_SIZE = 80;
const MAX_SIZE = 200;

const MicIcon: React.FC = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
  </svg>
);

const VoiceReactiveVisualizer: React.FC<VoiceReactiveVisualizerProps> = ({ isDarkMode, autoStart = false, audioSource = 'microphone', skipMicrophoneRequest = false, externalShowControls = false }) => {
  const [permissionState, setPermissionState] = useState<PermissionState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [blobs, setBlobs] = useState<Blob[]>([]);
  const [showControls, setShowControls] = useState(false);
  const [blobCount, setBlobCount] = useState(NUM_BLOBS);
  const [blurAmount, setBlurAmount] = useState(40);
  const [contrastAmount, setContrastAmount] = useState(30);
  const [sensitivity, setSensitivity] = useState(1.0);

  // Get the bot's audio track using the same method as PlasmaVisualizer
  const audioTrack = usePipecatClientMediaTrack("audio", "bot");

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const outputAnalyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const outputDataArrayRef = useRef<Uint8Array | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);
  const smoothedVolumeRef = useRef(0);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  
  const blobPhysicsRef = useRef<BlobPhysics[]>([]);
  const blobElementsRef = useRef<(HTMLDivElement | null)[]>([]);

  const createBlobs = useCallback(() => {
    const newBlobs: Blob[] = [];
    const newBlobPhysics: BlobPhysics[] = [];
    
    const colors = isDarkMode 
      ? ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FED766', '#F4A261']
      : ['#FF006E', '#FB5607', '#FFBE0B', '#8338EC', '#3A86FF'];
    
    for (let i = 0; i < blobCount; i++) {
        const id = i;
        const size = Math.random() * (MAX_SIZE - MIN_SIZE) + MIN_SIZE;
        newBlobs.push({
            id,
            size,
            color: colors[i % colors.length],
        });
        newBlobPhysics.push({
            id,
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            vx: (Math.random() - 0.5) * 2 * (MAX_SPEED - MIN_SPEED) + MIN_SPEED,
            vy: (Math.random() - 0.5) * 2 * (MAX_SPEED - MIN_SPEED) + MIN_SPEED,
        });
    }
    setBlobs(newBlobs);
    blobPhysicsRef.current = newBlobPhysics;
    blobElementsRef.current = new Array(newBlobs.length).fill(null);
  }, [blobCount, isDarkMode]);

  const animationLoop = useCallback(() => {
    if (!analyserRef.current || !dataArrayRef.current) {
        animationFrameIdRef.current = requestAnimationFrame(animationLoop);
        return;
    }

    // Get audio data
    analyserRef.current.getByteFrequencyData(dataArrayRef.current as Uint8Array);
    let sum = 0;
    for (const amplitude of dataArrayRef.current) {
      sum += amplitude * amplitude;
    }
    const rms = Math.sqrt(sum / dataArrayRef.current.length);
    let normalizedRms = Math.min(rms / 128, 1.0) * sensitivity;
    
    // Log audio levels occasionally for debugging
    if (Math.random() < 0.01) { // 1% chance to log
      // console.log('🎵 Audio level:', { rms: rms.toFixed(2), normalized: normalizedRms.toFixed(2) });
    }

    const SMOOTHING_FACTOR = 0.1;
    smoothedVolumeRef.current = SMOOTHING_FACTOR * normalizedRms + (1 - SMOOTHING_FACTOR) * smoothedVolumeRef.current;
    
    const volume = smoothedVolumeRef.current;
    const speedFactor = 1 + volume * 5;
    const scaleFactor = 1 + volume * 1.5;

    blobPhysicsRef.current = blobPhysicsRef.current.map(p => {
        let newX = p.x + p.vx * speedFactor;
        let newY = p.y + p.vy * speedFactor;

        const size = blobs.find(b => b.id === p.id)?.size || MAX_SIZE;
        if (newX > window.innerWidth + size / 2) newX = -size / 2;
        if (newX < -size / 2) newX = window.innerWidth + size / 2;
        if (newY > window.innerHeight + size / 2) newY = -size / 2;
        if (newY < -size / 2) newY = window.innerHeight + size / 2;
        
        return { ...p, x: newX, y: newY };
    });

    blobElementsRef.current.forEach((el, i) => {
        if (el) {
            const physics = blobPhysicsRef.current[i];
            if (physics) {
                el.style.transform = `translate(${physics.x}px, ${physics.y}px) scale(${scaleFactor})`;
            }
        }
    });

    animationFrameIdRef.current = requestAnimationFrame(animationLoop);
  }, [blobs, sensitivity]);

  // Set up audio analysis when we have an audio track
  useEffect(() => {
    if (!audioTrack) {
      console.log('🎵 No audio track available yet');
      return;
    }

    // console.log('🎵 Bot audio track received:', audioTrack);
    
    const setupAudio = async () => {
      try {
        const AudioContextClass = (window as typeof window & { webkitAudioContext?: typeof AudioContext }).AudioContext || 
                                 (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
        if (!AudioContextClass) {
          throw new Error('AudioContext not supported');
        }
        
        const context = new AudioContextClass();
        audioContextRef.current = context;

        // Create analyser
        const analyser = context.createAnalyser();
        analyser.fftSize = 256;
        analyserRef.current = analyser;
        dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);

        // Create media stream from audio track and connect to analyser
        const stream = new MediaStream([audioTrack]);
        const source = context.createMediaStreamSource(stream);
        source.connect(analyser);
        
        console.log('🎵 Successfully connected to bot audio track!');
        
        createBlobs();
        setPermissionState('granted');
      } catch (err) {
        console.error('🎵 Error setting up audio analysis:', err);
        setError('Could not analyze audio.');
        setPermissionState('denied');
      }
    };

    setupAudio();
    
    // Cleanup
    return () => {
      audioContextRef.current?.close().catch(console.error);
      audioContextRef.current = null;
    };
  }, [audioTrack, createBlobs]);

  const startVisualizer = useCallback(async () => {
    // Just start the blobs - audio setup is handled by the useEffect above
    if (permissionState !== 'granted') {
      setPermissionState('granted');
      createBlobs();
    }
  }, [createBlobs, permissionState]);

  const stopVisualizer = useCallback(() => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    audioContextRef.current?.close().catch(console.error);
    audioContextRef.current = null;
    analyserRef.current = null;
    dataArrayRef.current = null;
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }
    setPermissionState('idle');
    setBlobs([]);
    blobPhysicsRef.current = [];
    blobElementsRef.current = [];
  }, []);

  useEffect(() => {
    if (autoStart || skipMicrophoneRequest) {
      startVisualizer();
    }
  }, [autoStart, skipMicrophoneRequest, startVisualizer]);

  useEffect(() => {
    if (permissionState === 'granted') {
      animationFrameIdRef.current = requestAnimationFrame(animationLoop);
    }
    return () => {
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
    };
  }, [permissionState, animationLoop]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach(track => track.stop());
      audioContextRef.current?.close().catch(console.error);
    };
  }, []);

  useEffect(() => {
    const handleResize = () => {
        if (permissionState === 'granted') {
            createBlobs();
        }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [permissionState, createBlobs]);

  useEffect(() => {
    if (permissionState === 'granted' && blobs.length !== blobCount) {
      createBlobs();
    }
  }, [blobCount, createBlobs, permissionState, blobs.length]);

  // Only show permission UI if we're not skipping microphone request AND not granted
  if (!skipMicrophoneRequest && permissionState !== 'granted') {
    return (
      <div className="absolute inset-0 z-20 flex flex-col items-center justify-center p-4">
        <div className="text-center">
          {permissionState === 'idle' && (
            <>
              <h2 className={`text-2xl font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-black'}`}>
                Voice Reactive Mode
              </h2>
              <p className={`mb-6 max-w-md mx-auto ${isDarkMode ? 'text-white/80' : 'text-black/80'}`}>
                This visualizer reacts to your microphone to create a dynamic metaball effect.
              </p>
              <button 
                onClick={startVisualizer} 
                className={`
                  inline-flex items-center gap-3 px-6 py-3 font-semibold transition-all duration-300 
                  rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2
                  ${isDarkMode 
                    ? 'bg-white/90 text-black hover:bg-white focus:ring-offset-black focus:ring-white' 
                    : 'bg-black/90 text-white hover:bg-black focus:ring-offset-white focus:ring-black'}
                  hover:scale-[1.02] active:scale-[0.98] transform
                `}
              >
                <MicIcon/>
                Enable Microphone
              </button>
            </>
          )}
          {permissionState === 'pending' && (
            <p className={isDarkMode ? 'text-white' : 'text-black'}>
              Requesting microphone access...
            </p>
          )}
          {error && (
            <p className="text-red-400 max-w-md mx-auto">{error}</p>
          )}
        </div>
      </div>
    );
  }
  
  // If we haven't initialized yet but are skipping mic request, show loading
  if (skipMicrophoneRequest && permissionState === 'idle') {
    return (
      <div className="absolute inset-0 flex items-center justify-center">
        <div className={`animate-pulse ${isDarkMode ? 'text-white' : 'text-black'}`}>
          Initializing visualizer...
        </div>
      </div>
    );
  }

  return (
    <div className="absolute inset-0">
      {/* Blobs container with filter */}
      <div 
        className="absolute inset-0 overflow-hidden"
        style={{ 
          backgroundColor: isDarkMode ? 'black' : 'white',
          filter: `blur(${blurAmount}px) contrast(${contrastAmount})` 
        }}
      >
        {blobs.map((blob, i) => (
          <div
            key={blob.id}
            ref={el => { blobElementsRef.current[i] = el; }}
            className="absolute top-0 left-0 rounded-full will-change-transform"
            style={{
                width: `${blob.size}px`,
                height: `${blob.size}px`,
                backgroundColor: blob.color,
            }}
          />
        ))}
      </div>
      
      {/* Control Panel - Only show built-in button if NOT using external controls AND skipMicrophoneRequest is false */}
      {!skipMicrophoneRequest && !externalShowControls && (
        <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 z-40 pointer-events-auto">
          <button
            onClick={() => setShowControls(!showControls)}
            className={`
              px-4 py-2 rounded-full transition-all duration-300
              ${isDarkMode 
                ? 'bg-white/90 text-black hover:bg-white' 
                : 'bg-black/90 text-white hover:bg-black'}
              hover:scale-[1.02] active:scale-[0.98] transform
            `}
          >
            <span className="font-light text-xs tracking-[0.15em] uppercase">
              {showControls ? 'Hide' : 'Show'} Controls
            </span>
          </button>
        </div>
      )}

      {/* Show controls when either internal or external control is active */}
      {(showControls || externalShowControls) && (
        <div className={`
          fixed top-24 right-6 z-50
          px-6 py-4 rounded-lg backdrop-blur-md transition-all duration-300
          ${isDarkMode 
            ? 'bg-black/80 border border-white/20 text-white' 
            : 'bg-white/80 border border-black/20 text-black'}
          pointer-events-auto
        `}>
          <div className="space-y-4 min-w-[300px]">
            <div>
              <label className="text-xs font-light tracking-[0.15em] uppercase block mb-2">
                Blobs: {blobCount}
              </label>
              <input
                type="range"
                min="3"
                max="10"
                value={blobCount}
                onChange={(e) => setBlobCount(Number(e.target.value))}
                className="w-full cursor-pointer accent-current"
              />
            </div>
            
            <div>
              <label className="text-xs font-light tracking-[0.15em] uppercase block mb-2">
                Blur: {blurAmount}px
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={blurAmount}
                onChange={(e) => setBlurAmount(Number(e.target.value))}
                className="w-full cursor-pointer accent-current"
              />
            </div>
            
            <div>
              <label className="text-xs font-light tracking-[0.15em] uppercase block mb-2">
                Contrast: {contrastAmount}
              </label>
              <input
                type="range"
                min="1"
                max="50"
                value={contrastAmount}
                onChange={(e) => setContrastAmount(Number(e.target.value))}
                className="w-full cursor-pointer accent-current"
              />
            </div>
            
            <div>
              <label className="text-xs font-light tracking-[0.15em] uppercase block mb-2">
                Sensitivity: {sensitivity.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.1"
                max="3"
                step="0.1"
                value={sensitivity}
                onChange={(e) => setSensitivity(Number(e.target.value))}
                className="w-full cursor-pointer accent-current"
              />
            </div>
            
            <div>
              <label className="text-xs font-light tracking-[0.15em] uppercase block mb-2">
                Audio Source
              </label>
              <div className="text-xs opacity-70">
                {audioSource === 'microphone' && '🎤 Microphone Only'}
                {audioSource === 'output' && '🔊 TTS Output Only'}
                {audioSource === 'both' && '🎤+🔊 Both Sources'}
              </div>
            </div>

            {!skipMicrophoneRequest && (
              <button
                onClick={stopVisualizer}
                className={`
                  w-full px-4 py-2 rounded-full transition-all duration-300 mt-4
                  ${isDarkMode 
                    ? 'bg-red-500/90 text-white hover:bg-red-500' 
                    : 'bg-red-600/90 text-white hover:bg-red-600'}
                  hover:scale-[1.02] active:scale-[0.98] transform
                `}
              >
                <span className="font-light text-xs tracking-[0.15em] uppercase">
                  Stop Visualizer
                </span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default VoiceReactiveVisualizer;