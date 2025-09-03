#!/bin/bash
# Configure better models for M3 fact extraction
# Run this before starting the bot for improved extraction quality

echo "🚀 Configuring improved models for M3 fact extraction..."

# Set larger models for better understanding
export DSPY_REL_MODEL="qwen3-4b-instruct-2507:2"
export DSPY_FACTS_MODEL="qwen3-4b-instruct-2507"

# Alternative: Try even larger models if you have the RAM/VRAM
# export DSPY_REL_MODEL="qwen2.5-7b-instruct"
# export DSPY_FACTS_MODEL="qwen2.5-7b-instruct"

echo "✅ Model configuration:"
echo "   Relation extraction model: $DSPY_REL_MODEL"
echo "   Fact extraction model: $DSPY_FACTS_MODEL"
echo ""
echo "💡 These models provide better context understanding and reduce extraction noise."
echo "   Start the bot with: ./run_bot.sh"
echo ""
echo "🔧 To make this permanent, add these lines to your ~/.bashrc or ~/.zshrc:"
echo "   export DSPY_REL_MODEL=\"qwen2.5-3b-instruct\""
echo "   export DSPY_FACTS_MODEL=\"qwen2.5-3b-instruct\""