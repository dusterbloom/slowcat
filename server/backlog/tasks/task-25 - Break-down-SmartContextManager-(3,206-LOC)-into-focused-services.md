---
id: task-25
title: 'Break down SmartContextManager (3,206 LOC) into focused services'
status: To Do
assignee: []
created_date: '2025-09-01 21:35'
labels: []
dependencies: []
priority: high
---

## Description

Refactor the massive SmartContextManager into smaller, focused components while keeping functionality

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Extract TokenBudgetManager for token counting and allocation
- [ ] #2 Extract FactExtractor for background fact extraction
- [ ] #3 Extract SessionTracker for session metadata management
- [ ] #4 Extract DynamicPromptGenerator for context-aware prompts
- [ ] #5 Keep core SmartContextManager as orchestrator with clean interfaces
- [ ] #6 Maintain exactly 4096 token budget functionality
- [ ] #7 Ensure all integration tests still pass
<!-- AC:END -->
