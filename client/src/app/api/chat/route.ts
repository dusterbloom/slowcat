import { StreamingTextResponse, streamText } from 'ai';
import { openai } from '@ai-sdk/openai';

export const runtime = 'edge';

export async function POST(req: Request) {
  const { messages } = await req.json();

  // This is where we'll integrate with your Pipecat voice pipeline
  // For now, create a streaming response that your TTS can consume
  const result = await streamText({
    model: openai('gpt-4'),
    messages,
  });

  return new StreamingTextResponse(result.toAIStream());
}