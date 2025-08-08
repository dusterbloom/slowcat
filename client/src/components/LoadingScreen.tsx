"use client";

import { useState, useEffect } from 'react';

export function LoadingScreen({ onComplete }: { onComplete: () => void }) {
  const [progress, setProgress] = useState(0);
  const [dots, setDots] = useState('');
  
  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(onComplete, 500);
          return 100;
        }
        return prev + 2;
      });
    }, 50);
    
    return () => clearInterval(interval);
  }, [onComplete]);

  useEffect(() => {
    const dotsInterval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);
    
    return () => clearInterval(dotsInterval);
  }, []);
  
  return (
    <div className="fixed inset-0 bg-black flex items-center justify-center">
      <div className="text-center">
        <div className="mb-12">
          <div className="w-12 h-12 border border-white/20 border-t-white/60 rounded-full animate-spin mx-auto"></div>
        </div>
        
        <h1 className="font-light text-sm tracking-[0.3em] uppercase text-white mb-8">
          Slowcat
        </h1>
        
        <div className="w-64 h-[1px] bg-white/10 rounded-full overflow-hidden mx-auto relative">
          <div 
            className="absolute inset-y-0 left-0 bg-white/40 transition-all duration-100"
            style={{ width: `${progress}%` }}
          />
        </div>
        
        <p className="font-light text-[10px] tracking-[0.15em] uppercase text-white/40 mt-8">
          Initializing{dots}
        </p>
      </div>
    </div>
  );
}