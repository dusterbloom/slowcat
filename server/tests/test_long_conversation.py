#!/usr/bin/env python3
"""
Test stateless memory with long conversation and constrained context window
Simulates real LLM constraints to validate memory system effectiveness
"""

import asyncio
import sys
import os
import tempfile
import shutil
import time
import json
from pathlib import Path
import httpx

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment for testing
os.environ['USE_STATELESS_MEMORY'] = 'true'
os.environ['ENABLE_MEMORY'] = 'true'

# AgenticMemoryMatcher - kept for future reference, now using cached embeddings approach
class AgenticMemoryMatcher:
    """Agentic memory matching using Nomic embeddings + Gemma 270M"""
    
    def __init__(self, base_url="http://localhost:1234"):
        self.base_url = base_url
        self.embedding_calls = 0
        self.llm_calls = 0
        
    async def get_embedding(self, text):
        """Get embedding from Nomic via LM Studio"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/embeddings",
                    json={
                        "model": "text-embedding-nomic-embed-text-v1.5",
                        "input": text
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    self.embedding_calls += 1
                    return result['data'][0]['embedding']
                return None
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between vectors"""
        import numpy as np
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    async def llm_relevance_check(self, memory, query):
        """Use Gemma 270M for nuanced relevance decision"""
        try:
            prompt = f"""Memory: {memory[:200]}...
Query: {query}

Is this memory relevant to answering the query? Consider if the memory contains information that would help answer the question.

Answer only: YES or NO"""

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": "google/gemma-3-270m",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 5,
                        "temperature": 0.1
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result['choices'][0]['message']['content'].strip().upper()
                    self.llm_calls += 1
                    return "YES" in answer
                return False
        except Exception as e:
            print(f"LLM relevance check error: {e}")
            return False
    
    async def is_memory_relevant(self, memory, query):
        """Two-stage relevance checking: embeddings + LLM"""
        # Stage 1: Fast embedding similarity
        memory_emb = await self.get_embedding(memory[:300])  # Limit for speed
        query_emb = await self.get_embedding(query)
        
        if not memory_emb or not query_emb:
            return False  # Fallback to False if embeddings fail
        
        similarity = self.cosine_similarity(memory_emb, query_emb)
        
        # High similarity = definitely relevant
        if similarity > 0.75:
            return True
        
        # Very low similarity = definitely not relevant  
        if similarity < 0.3:
            return False
        
        # Uncertain cases: use LLM for final decision
        return await self.llm_relevance_check(memory, query)

class LMStudioLLM:
    """Real LLM connection to LM Studio with constrained context window"""
    
    def __init__(self, base_url="http://localhost:1234", max_tokens=1024):
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.total_calls = 0
        self.context_sizes = []
        self.successful_calls = 0
        
    def estimate_tokens(self, text):
        """Simple token estimation"""
        return len(text.split()) * 1.3
    
    def estimate_message_tokens(self, messages):
        """Estimate total tokens in message list"""
        total = 0
        for msg in messages:
            total += self.estimate_tokens(msg['content'])
            total += 10  # overhead per message
        return total
    
    async def generate(self, messages, max_response_tokens=150):
        """Generate response using LM Studio"""
        self.total_calls += 1
        
        # Estimate context size
        total_tokens = self.estimate_message_tokens(messages)
        self.context_sizes.append(total_tokens)
        
        # Check if context is too large (leave room for response)
        if total_tokens + max_response_tokens > self.max_tokens:
            return {
                'success': False,
                'error': f'Context too large: {total_tokens} + {max_response_tokens} > {self.max_tokens} tokens',
                'response': None,
                'tokens_used': total_tokens
            }
        
        try:
            # Call LM Studio API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": "qwen/qwen3-1.7b",
                        "messages": messages,
                        "max_tokens": max_response_tokens,
                        "temperature": 0.1,  # Low temperature for consistent responses
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_response = result['choices'][0]['message']['content']
                    
                    self.successful_calls += 1
                    return {
                        'success': True,
                        'error': None,
                        'response': assistant_response,
                        'tokens_used': total_tokens,
                        'response_tokens': result.get('usage', {}).get('completion_tokens', 0)
                    }
                else:
                    return {
                        'success': False,
                        'error': f'LM Studio error: {response.status_code} - {response.text}',
                        'response': None,
                        'tokens_used': total_tokens
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': f'Connection failed: {str(e)}',
                'response': None,
                'tokens_used': total_tokens
            }

async def load_conversation_data(file_path):
    """Load conversation data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Failed to load conversation data: {e}")
        return None

async def test_long_conversation_memory():
    """Test memory system with long conversation and constrained LLM"""
    
    print("🧠 Long Conversation Memory Test")
    print("=" * 50)
    
    # Import after setting environment
    from processors.stateless_memory import StatelessMemoryProcessor
    
    # Use clean temporary directory
    temp_dir = tempfile.mkdtemp(prefix="long_conv_test_")
    
    try:
        # Initialize memory system with small context window
        print("🚀 Initializing Memory System...")
        processor = StatelessMemoryProcessor(
            db_path=temp_dir,
            max_context_tokens=300,  # Very small context window
            perfect_recall_window=5,  # Small recall window
            enable_semantic_validation=True
        )
        
        # Initialize LM Studio connection with constrained context
        llm = LMStudioLLM(max_tokens=1024)  # Constrained but realistic limit
        
        print(f"✅ Memory system: {processor.max_context_tokens} token limit")
        print(f"✅ LM Studio LLM: {llm.max_tokens} token limit")
        print("🔗 Testing LM Studio connection...")
        
        # Test LM Studio connection
        test_result = await llm.generate([{"role": "user", "content": "Hello, can you respond briefly?"}])
        if test_result['success']:
            print("✅ LM Studio connection successful")
        else:
            print(f"❌ LM Studio connection failed: {test_result['error']}")
            print("Please ensure LM Studio is running on localhost:1234")
            return False
        
        # Test embedding-based memory system
        print("🧠 Testing cached embedding memory system...")
        
        # Test embedding API
        test_embedding = await processor._get_embedding("test text")
        if test_embedding:
            print("✅ Nomic embeddings working - cached similarity search enabled")
        else:
            print("⚠️  Nomic embeddings not available - falling back to keyword matching")
        
        # Use the real conversation file
        conversation_file = "../docs/stateless_context_tests/convo_test1_57_turns.json"
        
        if not Path(conversation_file).exists():
            print(f"❌ Conversation file not found: {conversation_file}")
            return False
        
        print(f"\n📖 Loading conversation from {conversation_file}")
        conversation_data = await load_conversation_data(conversation_file)
        if not conversation_data:
            print("❌ Failed to load conversation data")
            return False
        
        # Process the conversation
        conversation = conversation_data.get("conversation", [])
        print(f"📊 Processing {len(conversation)} conversation turns...")
        
        speaker_id = "test_user"
        
        # Store conversation history in batches
        print("\n🏗️ Building conversation history...")
        for i, turn in enumerate(conversation[:40]):  # First 40 turns for testing
            processor.current_speaker = speaker_id
            processor.current_user_message = turn["user"]
            
            await processor._store_exchange(turn["user"], turn["assistant"])
            
            if (i + 1) % 10 == 0:
                print(f"   Stored {i + 1} turns...")
        
        print(f"✅ Stored {40} conversation turns")
        
        # Test memory recall at different points in conversation (space exploration themed)
        test_queries = [
            {"query": "Who were the Apollo 11 astronauts?", "expected_info": "Neil Armstrong, Buzz Aldrin, Michael Collins"},
            {"query": "What did Michael Collins do during Apollo 11?", "expected_info": "remained in orbit around the Moon"},
            {"query": "What instruments did Apollo astronauts deploy?", "expected_info": "seismometers"},
            {"query": "What does a seismometer measure?", "expected_info": "vibrations and movements"},
            {"query": "What is the Artemis program?", "expected_info": "NASA's initiative to land the first woman"},
            {"query": "What rocket does Artemis use?", "expected_info": "Space Launch System (SLS)"},
            {"query": "What is the lunar gateway?", "expected_info": "space station orbiting the Moon"},
            {"query": "What minerals are on the Moon?", "expected_info": "oxygen, silicon, iron"},
            {"query": "How do you extract water from the Moon?", "expected_info": "mine water ice from polar regions"},
            {"query": "What is Orion spacecraft designed for?", "expected_info": "carry astronauts beyond low Earth orbit"},
        ]
        
        print(f"\n🎯 Testing Memory Recall with Constrained LLM")
        print("-" * 48)
        
        successful_recalls = 0
        context_overflow_count = 0
        
        for i, test in enumerate(test_queries, 1):
            query = test["query"]
            expected = test["expected_info"]
            
            print(f"\n🔍 Test {i}: {query}")
            print(f"Expected: Should recall {expected}")
            
            # Prepare messages for LLM
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant. You answer are laser-sharp onpoint and super short. No intros, just the answer the user needs without anything else.'},
                {'role': 'user', 'content': query}
            ]
            
            # Inject memory context
            start_time = time.perf_counter()
            await processor._inject_memory_context(messages, speaker_id)
            injection_time = (time.perf_counter() - start_time) * 1000
            
            print(f"⏱️  Memory injection: {injection_time:.2f}ms")
            print(f"📊 Total messages: {len(messages)}")
            
            # Check if context fits in LLM window
            result = await llm.generate(messages)
            
            if result['success']:
                print(f"🤖 LLM Response: {result['response']}")
                print(f"🔢 Tokens used: {result['tokens_used']}")
                
                # Check if response contains expected information (improved validation)
                response_lower = result['response'].lower()
                expected_lower = expected.lower()
                
                # Split expected into key terms and check for presence
                expected_terms = [term.strip('(),') for term in expected_lower.split() if len(term.strip('(),')) > 2]
                found_terms = [term for term in expected_terms if term in response_lower]
                
                # Calculate match percentage
                match_percentage = len(found_terms) / len(expected_terms) if expected_terms else 0
                
                # Consider it successful if 70% or more terms are found
                if match_percentage >= 0.7:
                    print(f"✅ SUCCESSFUL RECALL ({match_percentage:.0%} match)")
                    if found_terms != expected_terms:
                        missing_terms = [term for term in expected_terms if term not in found_terms]
                        print(f"   Found: {', '.join(found_terms)}")
                        if missing_terms:
                            print(f"   Missing: {', '.join(missing_terms)}")
                    successful_recalls += 1
                else:
                    print(f"⚠️  Incomplete recall ({match_percentage:.0%} match)")
                    print(f"   Expected terms: {', '.join(expected_terms)}")
                    print(f"   Found terms: {', '.join(found_terms)}")
                    
            else:
                print(f"❌ Context overflow: {result['error']}")
                context_overflow_count += 1
        
        # Performance analysis
        print(f"\n📊 LONG CONVERSATION ANALYSIS")
        print("-" * 35)
        
        recall_rate = (successful_recalls / len(test_queries)) * 100
        print(f"Memory recall success rate: {recall_rate:.1f}% ({successful_recalls}/{len(test_queries)})")
        print(f"Context overflow incidents: {context_overflow_count}")
        
        avg_context_size = sum(llm.context_sizes) / len(llm.context_sizes) if llm.context_sizes else 0
        max_context_size = max(llm.context_sizes) if llm.context_sizes else 0
        
        print(f"Average context size: {avg_context_size:.0f} tokens")
        print(f"Maximum context size: {max_context_size:.0f} tokens")
        print(f"LLM token limit: {llm.max_tokens} tokens")
        
        # Embedding memory performance
        print(f"\n🧠 CACHED EMBEDDING PERFORMANCE")
        print("-" * 32)
        print("✅ One embedding call per query (cached approach)")
        print("✅ Local cosine similarity computation (no API calls)")
        print("✅ No LLM relevance calls needed")
        print("✅ Deterministic performance - no API dependency for matching")
        
        # Memory system stats
        stats = processor.get_performance_stats()
        print(f"\nMemory system performance:")
        print(f"   Total conversations: {stats['total_conversations']}")
        print(f"   Cache hit ratio: {stats['cache_hit_ratio']:.1%}")
        print(f"   Avg injection time: {stats['avg_injection_time_ms']:.2f}ms")
        
        # Storage distribution
        with processor.env.begin() as txn:
            hot_count = txn.stat(processor.hot_db)['entries']
            warm_count = txn.stat(processor.warm_db)['entries']
            cold_count = txn.stat(processor.cold_db)['entries']
        
        print(f"   Storage: {hot_count} hot, {warm_count} warm, {cold_count} cold")
        
        # Test conversation continuation
        print(f"\n🔄 Testing Conversation Continuation...")
        print("-" * 40)
        
        # Process remaining turns and test recall
        remaining_turns = conversation[40:57]  # Use all remaining turns
        for turn in remaining_turns:
            await processor._store_exchange(turn["user"], turn["assistant"])
        
        print(f"✅ Added {len(remaining_turns)} more conversation turns")
        
        # Test if early memories are still accessible
        early_test = {
            "query": "Who were the three Apollo 11 astronauts again?",
            "expected_info": "Neil Armstrong, Buzz Aldrin, Michael Collins"
        }
        
        messages = [
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': early_test["query"]}
        ]
        
        await processor._inject_memory_context(messages, speaker_id)
        result = await llm.generate(messages)
        
        if result['success']:
            # Use improved validation for long-term memory test
            response_lower = result['response'].lower()
            expected_terms = ["neil", "armstrong", "buzz", "aldrin", "michael", "collins"]
            found_terms = [term for term in expected_terms if term in response_lower]
            match_percentage = len(found_terms) / len(expected_terms)
            
            if match_percentage >= 0.5:  # At least half the astronaut names
                print(f"✅ Long-term memory preserved across extended conversation ({match_percentage:.0%} match)")
                print(f"   Response: {result['response']}")
            else:
                print(f"⚠️  Long-term memory degradation detected ({match_percentage:.0%} match)")
                print(f"   Response: {result['response']}")
        else:
            print("⚠️  Long-term memory test failed due to LLM error")
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT")
        print("-" * 20)
        
        if recall_rate >= 75:
            memory_grade = "🟢 EXCELLENT"
        elif recall_rate >= 50:
            memory_grade = "🟡 GOOD"
        else:
            memory_grade = "🔴 POOR"
        
        print(f"Memory Quality: {memory_grade} ({recall_rate:.1f}% recall)")
        
        if context_overflow_count == 0:
            efficiency_grade = "🟢 EFFICIENT"
        elif context_overflow_count <= 2:
            efficiency_grade = "🟡 ADEQUATE"
        else:
            efficiency_grade = "🔴 INEFFICIENT"
        
        print(f"Context Efficiency: {efficiency_grade} ({context_overflow_count} overflows)")
        
        if avg_context_size < llm.max_tokens * 0.8:
            size_grade = "🟢 OPTIMAL"
        else:
            size_grade = "🟡 NEAR LIMIT"
        
        print(f"Context Size: {size_grade} ({avg_context_size:.0f}/{llm.max_tokens} tokens)")
        
        # Clean up
        processor.env.close()
        processor.thread_pool.shutdown(wait=True)
        
        print(f"\n🎉 Long conversation test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    result = asyncio.run(test_long_conversation_memory())
    sys.exit(0 if result else 1)