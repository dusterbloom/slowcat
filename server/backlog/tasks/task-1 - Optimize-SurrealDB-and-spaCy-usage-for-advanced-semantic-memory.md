---
id: task-1
title: Optimize SurrealDB and spaCy usage for advanced semantic memory
status: To Do
assignee: []
created_date: '2025-08-28 15:09'
labels:
  - memory
  - optimization
  - surrealdb
  - spacy
dependencies: []
priority: high
---

## Description

Currently we're only using ~30% of SurrealDB and spaCy capabilities. We're treating SurrealDB like traditional SQL instead of leveraging graph relationships, and missing advanced spaCy features like coreference resolution and temporal extraction. This limits information retrieval effectiveness.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Refactor facts storage to use SurrealDB RELATE statements for graph relationships (user->owns->pet)
- [ ] #2 Implement SurrealDB built-in vector search instead of manual embedding similarity
- [ ] #3 Add spaCy coreference resolution to link pronouns to entities across sentences
- [ ] #4 Add temporal expression extraction for events, meetings, and dates
- [ ] #5 Create entity resolution system to track same entities across conversations
- [ ] #6 Implement graph traversal queries for multi-hop reasoning
- [ ] #7 Add custom domain NER for pets, locations, and user preferences
- [ ] #8 Test improved information retrieval with complex queries (meetings, relationships)
<!-- AC:END -->
