"use client";

import {
  ConsoleTemplate,
  FullScreenContainer,
  ThemeProvider,
} from "@pipecat-ai/voice-ui-kit";

export default function Home() {
  // Use the same ENABLE_VIDEO env variable as server
  const videoEnabled = process.env.NEXT_PUBLIC_ENABLE_VIDEO === "true";
  
  return (
    <ThemeProvider>
      <FullScreenContainer>
        <ConsoleTemplate
          transportType="smallwebrtc"
          connectParams={{
            connectionUrl: "/api/offer",
          }}
          noUserVideo={!videoEnabled}
        />
      </FullScreenContainer>
    </ThemeProvider>
  );
}
