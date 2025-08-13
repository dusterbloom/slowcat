"use client";

import { useState, useEffect } from 'react';
import { ThemeProvider } from "@pipecat-ai/voice-ui-kit";
import { setupLinkConversion } from '../utils/linkFormatter.js';
import { LoadingScreen } from '../components/LoadingScreen';
import { VoiceApp } from '../components/VoiceApp';

export default function Home() {
  const [isLoading, setIsLoading] = useState(true);
  const videoEnabled = process.env.NEXT_PUBLIC_ENABLE_VIDEO === "false" ? false : true;
  
  // Setup automatic markdown link conversion on mount
  useEffect(() => {
    // Add debugging to see what's happening
    console.log('🔍 Page mounted, setting up link conversion...');
    
    // Add a delay to let the voice UI load first
    const timer = setTimeout(() => {
      console.log('🔍 DOM elements:', document.body.innerHTML.length);
      console.log('🔍 Looking for text elements...');
      
      // Check what elements exist
      const allDivs = document.querySelectorAll('div');
      console.log('🔍 Found', allDivs.length, 'div elements');
      
      allDivs.forEach((div, index) => {
        if (index < 10 && div.textContent && div.textContent.includes('restaurant')) {
          console.log('🔍 Found restaurant text in div:', div.textContent.substring(0, 100));
          console.log('🔍 Div classes:', div.className);
          console.log('🔍 Div attributes:', [...div.attributes].map(a => `${a.name}="${a.value}"`));
        }
      });
      
      const observer = setupLinkConversion();
      
      // Manual test - try to find and convert any existing text
      const testElements = document.querySelectorAll('*');
      let foundLinks = 0;
      testElements.forEach(el => {
        const text = el.textContent || '';
        if (text.includes('Visit website') || (text.includes('[') && text.includes(']('))) {
          foundLinks++;
          console.log('🔗 Found linkable text:', text.substring(0, 100));
          console.log('🔗 Element:', el.tagName, el.className);
        }
      });
      console.log('🔗 Total elements with linkable text:', foundLinks);
      
      return () => observer?.disconnect();
    }, 2000);
    
    return () => clearTimeout(timer);
  }, []);
  
  if (isLoading) {
    return <LoadingScreen onComplete={() => setIsLoading(false)} />;
  }
  
  return (
    <ThemeProvider>
      <VoiceApp videoEnabled={videoEnabled} />
    </ThemeProvider>
  );
}
