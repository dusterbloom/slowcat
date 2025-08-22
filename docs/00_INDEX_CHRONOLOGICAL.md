# SlowCat Documentation Index (Chronological)

*Last Updated: 2025-08-21*

## Documentation Organization

All docs are now numbered for chronological tracking. Format: `XX_NAME.md` where XX is the order of creation/importance.

## Core Documentation (00-19)

- `00_INDEX_CHRONOLOGICAL.md` - This index file
- `01_QUICK_START.md` - Getting started guide
- `02_API_REFERENCE.md` - API documentation
- `03_WHATS_NEW.md` - Latest updates and changelog

## Memory & Architecture Evolution (20-39)

- `20_Journey_to_stateless_agent.md` - The philosophical journey to stateless memory
- `21_memory_implementation_strategy.md` - Initial memory strategy
- `22_stateless_memory_architecture_fix.md` - Stateless architecture improvements
- `23_stateless_a_mem_plan.md` - Advanced memory planning
- `24_stateless_llm_context_tools.md` - LLM context tool integration
- `25_AGENTIC_MEMORY_INTEGRATION_PLAN.md` - Agentic memory roadmap
- `26_TASK_99_FIXED_CONTEXT_SMART_MEMORY.md` - Fixed context smart memory implementation
- `27_SlowcatMemory_Analysis_Report.md` - Memory system analysis
- `28_MEMORY_TROUBLESHOOTING.md` - Memory debugging guide
- `29_first_real_test_bm25_lz4.md` - BM25 and LZ4 compression testing

## Integration Guides (40-59)

- `40_SURREALDB_INTEGRATION.md` - SurrealDB multi-model database integration
- `41_HANDOFF_SURREALDB.md` - SurrealDB handoff documentation
- `42_MCP_INTEGRATION_GUIDE.md` - Model Context Protocol integration
- `43_MCP_INTEGRATION_PLAN.md` - MCP integration planning
- `44_MCP_PIPECAT_INTEGRATION.md` - MCP with PipeCat pipeline
- `45_MCP_SETUP.md` - MCP setup instructions
- `46_MCP_SUMMARY.md` - MCP implementation summary
- `47_DEBUG_MCP_INTEGRATION.md` - MCP debugging guide

## Feature Documentation (60-79)

- `60_SHERPA_ONNX_STT.md` - Sherpa-ONNX STT implementation
- `61_NEW_CHATTERBOX_TTS.md` - Chatterbox TTS integration
- `62_BRAVE_WEB_SEARCH_DOCS.md` - Brave web search integration
- `63_FREE_WEB_SEARCH_GUIDE.md` - Free web search implementation
- `64_OFFLINE_MODE.md` - Offline mode operation
- `65_METAL_GPU_FIX.md` - macOS Metal GPU optimization

## Development & Analysis (80-89)

- `80_REFACTORING_GUIDE.md` - Code refactoring guidelines
- `81_PERFORMANCE_ANALYSIS.md` - Performance metrics and analysis
- `82_logs_server_turns_and_forgetting.md` - Server logging and memory decay

## External References (90-99)

- `90_pipecat-ai-docs-8a5edab282632443.txt` - PipeCat AI documentation
- `91_pipecat-ai-voice-ui-kit-8a5edab282632443.txt` - PipeCat Voice UI Kit
- `92_lmstudio-ai-docs-8a5edab282632443.txt` - LMStudio AI documentation

## Tomorrow's Work (100+)

- `100_TODO_DYNAMIC_TAPE_HEAD.md` - Dynamic Tape Head implementation plan
- `101_DTH_IMPLEMENTATION.md` - Dynamic Tape Head actual implementation (2025-08-21)
- `102_VOICE_MOOD_CAPTURE_NOTES.md` - Voice mood capture notes + safe stash plan (2025-08-21)

## Log Files

- `2025-08-06.3.log` - Historical log file
- `logs/` - Directory containing detailed logs
- `stateless_context_tests/` - Test results for stateless context

---

## Navigation Tips

1. **New to SlowCat?** Start with `01_QUICK_START.md`
2. **Understanding the architecture?** Read docs 20-29 in order
3. **Implementing features?** Check relevant integration guides (40-59)
4. **Tomorrow's work?** See `100_TODO_DYNAMIC_TAPE_HEAD.md`

## Contributing

When adding new documentation:
1. Use the next available number in the appropriate range
2. Update this index
3. Add creation date in the document header
4. Link related documents
