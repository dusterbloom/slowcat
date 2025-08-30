# Backlog: Implementing Advanced, Embedding-Based Memory Search

This document outlines the planned code changes to upgrade Slowcat's memory system from a keyword-based search to an advanced, hybrid search model leveraging vector embeddings.

---

## **Phase 1: Embedding Generation in Fact Extractor**

**Goal:** Modify the fact extraction pipeline to generate a vector embedding for every new fact.

**File to Modify:** `server/memory/hybrid_fact_extractor.py`

### **Task 1.1: Integrate Sentence Transformer Model**

-   **Action:** Load the `all-MiniLM-L6-v2` sentence transformer model upon initialization of the `HybridFactExtractor`. The implementation will prefer the `mlx_sentence_transformers` library for performance on Apple Silicon, with a fallback to the standard `sentence-transformers` library.
-   **Rationale:** This provides the core capability to convert textual facts into semantic vector embeddings.

### **Task 1.2: Update Data Structure to Hold Embeddings**

-   **Action:** Add an `embedding: Optional[List[float]] = None` field to the `HybridFact` dataclass.
-   **Rationale:** To store the generated vector within the fact object before it's passed to the database layer.

    ```python
    # Proposed change in HybridFact dataclass
    @dataclass
    class HybridFact:
        subject: str
        predicate: str
        value: str
        # ... other fields
        embedding: Optional[List[float]] = None # <-- ADD THIS

        def to_dict(self) -> Dict[str, Any]:
            # ...
            data['embedding'] = self.embedding # <-- ADD THIS
            return data
    ```

### **Task 1.3: Generate and Attach Embeddings**

-   **Action:** In the `_extract_facts_async` method, after facts are created, add a new step to generate embeddings.
-   **Logic:**
    1.  For each `HybridFact` object, create a descriptive string (e.g., `f"{fact.subject} {fact.predicate.replace('_', ' ')} {fact.value}"`).
    2.  Use the loaded sentence transformer model to encode this string into a vector.
    3.  Assign the resulting vector to the `fact.embedding` field.
-   **Rationale:** This enriches each fact with a semantic representation that the database can use for similarity searches.

---

## **Phase 2: Storing Embeddings in the Knowledge Graph**

**Goal:** Update the database schema and connection logic to persist fact embeddings.

### **Task 2.1: Update SurrealDB Schema**

-   **File to Modify:** `server/schema/hybrid_unified_schema.surql`
-   **Action:** Add an `embedding` field to the `knowledge` relation table.
-   **Rationale:** To create a column in the database for storing the fact embeddings.

    ```surql
    -- Proposed change in knowledge table definition
    DEFINE TABLE knowledge TYPE RELATION IN entity OUT entity SCHEMAFULL;
    DEFINE FIELD predicate ON knowledge TYPE string;
    -- ... other fields ...
    DEFINE FIELD embedding ON knowledge TYPE option<array<float>> FLEXIBLE; -- <-- ADD THIS
    ```

### **Task 2.2: Modify Database Connection to Store Embeddings**

-   **File to Modify:** `server/memory/surreal_connection.py`
-   **Action:** Update the `store_knowledge_relation` method to accept and store the embedding.
-   **Logic:**
    1.  The method signature will be updated to accept an optional `embedding` parameter.
    2.  The `RELATE ... SET` query will be modified to include `embedding = $embedding` if an embedding is provided.
-   **Rationale:** This completes the write path, ensuring that generated embeddings are correctly saved into the knowledge graph.

---

## **Phase 3: Implementing Advanced Search and Retrieval**

**Goal:** Replace the basic keyword search with the advanced, hybrid search function and update the client-side code to use it.

### **Task 3.1: Implement Advanced Search Function in SurrealDB**

-   **File to Modify:** `server/schema/hybrid_unified_schema.surql`
-   **Action:** Define the new `fn::search_knowledge_advanced` function as previously sketched.
-   **Rationale:** This moves the complex logic of hybrid search (combining full-text, vector similarity, graph proximity, and memory strength) into the database for maximum performance and efficiency.

### **Task 3.2: Implement Client-Side Query Embedding**

-   **File to Modify:** `server/memory/surreal_connection.py`
-   **Action:**
    1.  Ensure a sentence transformer model is accessible within the `SurrealConnectionManager` (or globally).
    2.  In the `search_knowledge_relations` method, before calling the database, generate a vector embedding from the user's text query.
-   **Rationale:** The advanced search function requires both the query text and its vector representation to perform the hybrid search.

### **Task 3.3: Update Search Method to Call the New Function**

-   **File to Modify:** `server/memory/surreal_connection.py`
-   **Action:** The `search_knowledge_relations` method will be updated to call `fn::search_knowledge_advanced` instead of the old function, passing the query text and the newly generated query vector.
-   **Rationale:** This activates the new advanced search capability, completing the read path. All components that rely on this method will now benefit from the improved search relevance.
