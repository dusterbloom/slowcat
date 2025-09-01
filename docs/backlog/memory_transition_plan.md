# Memory Architecture Transition Plan

**Objective:** To incrementally transition the agent's pipeline from a parallel context system to a stable, single-source-of-truth hybrid architecture. This plan leverages the stability of Pipecat's native context aggregator while enhancing it with the sophisticated long-term memory capabilities of the `SmartContextManager`.

---

### Current State Analysis (Post-Fix)

After re-enabling `context_aggregator.assistant()`, the pipeline's stability has significantly improved, eliminating the primary cause of duplicate assistant messages.

- **What Works:** The `OpenAILLMContextAggregator` now correctly maintains a clean, alternating turn history for the immediate session, preventing conversational "stutter."
- **The Remaining Gap:** The rich, long-term context prepared by `SmartContextManager` is not yet being injected into the LLM prompt. The agent is coherent turn-to-turn but is not yet leveraging its deep memory. We are running two parallel, disconnected context systems.

---

### The 3-Step Transition Plan

This plan will merge the two systems safely, with clear verification steps at each stage.

#### Step 1: The "Observer" Phase - Passively Prepare Context

**Goal:** Modify `SmartContextManager` to prepare the enhanced context but only log it for verification, without affecting the live pipeline. This is a safe, no-risk change to confirm the retrieval logic is sound.

**Actions:**
1.  In `server/processors/smart_context_manager.py`, modify the `process_frame` method.
2.  Keep all existing logic for querying SurrealDB and retrieving the "context package" (attractors, summaries, etc.).
3.  At the end of the method, construct the `enhanced_content` string that combines the user's live speech with the retrieved memories.
    ```python
    # Example logic
    enhanced_content = f"User Query: '{frame.text}'\n\n--- Relevant Memories ---
{retrieved_memories}"
    ```
4.  Log this content for verification: `logger.debug(f"Prepared Enhanced Context: {enhanced_content}")`.
5.  Allow the original `TranscriptionFrame` to pass through the pipeline unchanged.

**Verification:** Run the agent and have a conversation. The agent's behavior should not change. Review the logs to confirm that the detailed, memory-enhanced context is being correctly prepared for every user turn.

---

#### Step 2: The "Injection" Phase - Activate Manual Context Control

**Goal:** Activate the new architecture. The enhanced context from `SmartContextManager` will now be injected directly into the LLM's prompt, making the agent's long-term memory active.

**Actions:**
1.  In `SmartContextManager`'s `process_frame` method, create and `push` the `LLMUserContextFrame` containing the `enhanced_content`.
    ```python
    from pipecat.frames.frames import LLMUserContextFrame
    
    context_frame = LLMUserContextFrame(
        context={
            "role": "user",
            "content": enhanced_content
        }
    )
    await self.push(context_frame)
    
    # Ensure the original TranscriptionFrame is NOT passed on.
    # We are replacing it with our enhanced version.
    ```
2.  In `server/core/pipeline_builder.py`, ensure the `smart_ctx` processor is positioned immediately before `context_aggregator.user()`.

**Verification:** Run the agent. The agent's responses should now be qualitatively different. It should begin to reference past conversations and demonstrate long-term memory recall, confirming that the enhanced context is successfully being passed to the LLM.

---

#### Step 3: The "Cleanup" Phase - Finalize and Synchronize

**Goal:** Solidify the new architecture by synchronizing the `user_id` across all components, ensuring data integrity and completing the transition.

**Actions:**
1.  **Create a `update_user_id` method** in the `SmartContextManager` class. This method will update the internal `user_id` used for all its database operations.
2.  **Modify the `on_speaker_changed` callback** in `server/core/pipeline_builder.py`. This callback must now update the `user_id` on **both** the `memory_processor` and the `smart_ctx` instance.
    ```python
    # Example callback logic
    async def on_speaker_changed(data: Dict[str, Any]):
        user_id = data.get('speaker_name', data.get('speaker_id', 'unknown'))
        
        # Propagate the ID to both memory components
        if memory_processor:
            await memory_processor.update_user_id(user_id)
        if smart_ctx: # Assuming smart_ctx is accessible
            await smart_ctx.update_user_id(user_id)
            
        logger.info(f"📝 Memory and Context switched to user: {user_id}")
    ```
3.  **Review and Remove Redundancy:** With the new system in place, review the base system prompt and other configurations to remove any logic that is now handled by the dynamic context injection.

**Final Verification:** The system is now fully transitioned. It should be stable with no message duplication. The agent should demonstrate robust long-term memory. A database inspection should confirm that both user and assistant messages for a given session are correctly associated with the same, dynamically recognized `user_id`.
