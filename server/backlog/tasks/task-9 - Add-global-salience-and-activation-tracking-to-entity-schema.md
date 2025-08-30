---
id: task-9
title: Add global salience and activation tracking to entity schema
status: To Do
assignee: []
created_date: '2025-08-30 14:54'
labels:
  - consciousness
  - schema
  - surrealdb
dependencies: []
priority: high
---

## Description

Extend SurrealDB entity table with global_salience and last_activation fields to support dynamic symbol salience tracking as required by biological key-value memory architecture

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Entity table has global_salience field (float, default 0.0)
- [ ] #2 Entity table has last_activation field (datetime, default time::now())
- [ ] #3 Schema migration preserves existing entity data
- [ ] #4 All existing entity operations continue to work unchanged
<!-- AC:END -->
