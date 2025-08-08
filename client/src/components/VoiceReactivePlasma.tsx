"use client";

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

  if (!PlasmaVisualizer) {
    return (
      <div className={`absolute inset-0 animate-pulse ${
        isDarkMode 
          ? 'bg-gradient-to-br from-orange-900/20 to-purple-900/20' 
          : 'bg-gradient-to-br from-orange-200/50 to-yellow-200/50'
      }`} />
    );
  }

  return (
    <div className={`absolute inset-0 ${isDarkMode ? 'bg-black plasma-dark' : 'bg-white plasma-light'}`}>
      <PlasmaVisualizer 
        state="connected"
        style={{
          backgroundColor: isDarkMode ? 'black' : 'white'
        }}
      />
    </div>
  );
}