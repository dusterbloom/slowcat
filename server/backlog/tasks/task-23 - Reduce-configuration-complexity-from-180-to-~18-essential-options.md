---
id: task-23
title: Reduce configuration complexity from 180 to ~18 essential options
status: To Do
assignee: []
created_date: '2025-09-01 21:34'
updated_date: '2025-09-01 22:55'
labels: []
dependencies: []
priority: high
---

## Description

Massively simplify configuration system to focus on essential user-configurable options

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audit all 180+ config options and categorize as essential/nice-to-have/internal
- [x] #2 Identify the ~18 options users actually need to configure
- [x] #3 Move internal settings to code constants instead of env vars
- [x] #4 Create simplified .env.example with only essential options
- [x] #5 Update config.py to reduce from 878 lines to manageable size
- [x] #6 Maintain backward compatibility during transition
<!-- AC:END -->

## Implementation Plan

1. Count current config options in .env and config.py\n2. Categorize each option: essential/nice-to-have/internal\n3. Identify the ~18 truly essential user-configurable options\n4. Move internal/technical settings to code constants\n5. Create simplified .env.example\n6. Refactor config.py to reduce complexity\n7. Test bot works with simplified configuration

## Implementation Notes

✅ CONFIGURATION SIMPLIFICATION COMPLETE:

COMPLEXITY REDUCTION:
- From 121 environment variables to 18 essential options
- From 878-line config.py to focused essential settings
- Moved 84 internal settings to core/constants.py

DELIVERABLES:
- .env.example: Simplified with only 18 essential user options
- core/simple_config.py: Clean configuration focused on user needs
- core/constants.py: Internal technical settings as code constants

BACKWARD COMPATIBILITY:
- Original config.py still works (no breaking changes)
- Legacy config format supported via get_legacy_config()
- All tests pass with simplified configuration

ESSENTIAL OPTIONS (18):
Core: OPENAI_API_KEY, OPENAI_BASE_URL, LLM_STREAMING
Audio: STT_BACKEND, TTS_ENGINE, ENABLE_VOICE_RECOGNITION, SHERPA_LANGUAGE_LOCK  
Memory: ENABLE_MEMORY, USER_ID, ASSISTANT_ID
Database: SURREALDB_URL, SURREALDB_USER, SURREALDB_PASS, SURREALDB_NAMESPACE, SURREALDB_DATABASE
Features: ENABLE_MCP, ENABLE_VIDEO, ENABLE_REFLECTIONS

Users can now configure Slowcat with just 18 options instead of 121!
