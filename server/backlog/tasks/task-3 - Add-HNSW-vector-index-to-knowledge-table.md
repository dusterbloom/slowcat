---
id: task-3
title: Add HNSW vector index to knowledge table
status: To Do
assignee: []
created_date: '2025-08-30 13:13'
labels:
  - vector-search
  - database
  - performance
dependencies: []
priority: high
---

## Description

Implement HNSW (Hierarchical Navigable Small World) index on the knowledge table's embedding field to enable high-performance vector similarity search using SurrealDB's native capabilities

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 HNSW index is defined on knowledge.embedding field with DIMENSION 384 and COSINE distance,Index creation is verified through SurrealDB info query,Index performance is validated with sample vector queries,Index configuration matches embedding model dimensions (384 for sentence-transformers)
<!-- AC:END -->
