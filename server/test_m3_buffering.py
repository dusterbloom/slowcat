#!/usr/bin/env python3
"""
Test M3 Conversation Buffering System
Tests the complete M3 buffering pipeline: buffer -> chunk -> extraction -> storage
"""

import asyncio
import os
import time
from typing import List, Dict, Any
from loguru import logger

# Import M3 components
from memory.m3_conversation_buffer import M3ConversationBuffer
from memory.dspy_integration import extract_facts_from_chunk_dspy


class M3BufferingTester:
    """Test M3 conversation buffering end-to-end"""
    
    def __init__(self):
        self.test_conversations = [
            # Conversation 1: Job and personal info (should trigger after 3 turns)
            [
                ("I work at Google as a software engineer", "peppi"),
                ("I'm really passionate about AI and machine learning", "peppi"),
                ("My dog's name is Potola and she's a golden retriever", "peppi"),
                ("We love hiking together on weekends", "peppi"),
                ("She's really good at catching frisbees", "peppi")
            ],
            # Conversation 2: Food preferences and lifestyle
            [
                ("I love Italian food, especially pizza", "peppi"),
                ("I usually cook at home during weekdays", "peppi"),
                ("My favorite restaurant is this little place in North Beach", "peppi")
            ]
        ]
    
    def test_basic_buffer_functionality(self):
        """Test basic buffer functionality"""
        print("🧪 Testing basic buffer functionality...")
        
        buffer = M3ConversationBuffer(max_turns=3, max_seconds=30.0)
        
        # Test single turn
        should_process = buffer.add_turn("First message", "peppi")
        assert not should_process, "Should not process after 1 turn"
        
        # Test second turn
        should_process = buffer.add_turn("Second message", "peppi")
        assert not should_process, "Should not process after 2 turns"
        
        # Test third turn (should trigger)
        should_process = buffer.add_turn("Third message", "peppi")
        assert should_process, "Should process after 3 turns"
        
        # Test chunk creation
        chunk = buffer.get_chunk()
        assert chunk is not None, "Chunk should be created"
        assert chunk.total_turns == 3, f"Expected 3 turns, got {chunk.total_turns}"
        assert "First message" in chunk.combined_text, "Should contain first message"
        assert "Third message" in chunk.combined_text, "Should contain third message"
        
        print("✅ Basic buffer functionality working")
    
    def test_chunk_extraction_quality(self):
        """Test fact extraction quality from conversation chunks vs individual messages"""
        print("\n🧪 Testing chunk vs individual extraction quality...")
        
        # Test conversation: related statements about work and interests
        conversation_turns = [
            "I work at Google as a software engineer",
            "I'm really passionate about AI and machine learning",
            "My dog's name is Potola and she's a golden retriever"
        ]
        
        # Test 1: Individual message extraction
        individual_facts = []
        for turn in conversation_turns:
            try:
                from memory.dspy_integration import extract_facts_from_text_dspy
                facts = extract_facts_from_text_dspy(turn)
                individual_facts.extend(facts)
            except Exception as e:
                print(f"Individual extraction error: {e}")
        
        # Test 2: Chunk-based extraction
        chunk_text = " ".join(conversation_turns)
        try:
            chunk_facts = extract_facts_from_chunk_dspy(
                chunk_text=chunk_text,
                previous_context=""
            )
        except Exception as e:
            print(f"Chunk extraction error: {e}")
            chunk_facts = []
        
        # Compare results
        print(f"📊 Individual extraction: {len(individual_facts)} facts")
        for i, fact in enumerate(individual_facts, 1):
            print(f"   {i}. {fact.get('subject', 'N/A')} {fact.get('predicate', 'N/A')} {fact.get('value', 'N/A')} (conf: {fact.get('confidence', 0):.2f})")
        
        print(f"📊 Chunk extraction: {len(chunk_facts)} facts")
        for i, fact in enumerate(chunk_facts, 1):
            print(f"   {i}. {fact.get('subject', 'N/A')} {fact.get('predicate', 'N/A')} {fact.get('value', 'N/A')} (conf: {fact.get('confidence', 0):.2f})")
        
        # Analysis
        if len(chunk_facts) >= len(individual_facts):
            print("✅ Chunk extraction produced more or equal facts")
        else:
            print("⚠️ Chunk extraction produced fewer facts")
        
        return {
            'individual_facts': len(individual_facts),
            'chunk_facts': len(chunk_facts),
            'chunk_advantage': len(chunk_facts) - len(individual_facts)
        }
    
    def test_conversation_processing(self):
        """Test processing complete conversations"""
        print("\n🧪 Testing complete conversation processing...")
        
        results = []
        
        for conv_idx, conversation in enumerate(self.test_conversations, 1):
            print(f"\n📝 Testing conversation {conv_idx} ({len(conversation)} turns)")
            
            # Create buffer for this conversation
            buffer = M3ConversationBuffer(max_turns=3, max_seconds=30.0)
            chunks_processed = 0
            total_facts = 0
            
            for turn_idx, (text, speaker) in enumerate(conversation, 1):
                print(f"   Turn {turn_idx}: '{text[:50]}...' (speaker: {speaker})")
                
                should_process = buffer.add_turn(text, speaker)
                status = buffer.get_buffer_status()
                
                print(f"   Buffer: {status['turns_count']} turns, should_process: {should_process}")
                
                if should_process:
                    # Process chunk
                    chunk = buffer.get_chunk()
                    if chunk:
                        chunks_processed += 1
                        print(f"\n🔄 PROCESSING CHUNK {chunks_processed}:")
                        print(f"   Turns: {chunk.total_turns}")
                        print(f"   Text: '{chunk.combined_text[:100]}...'")
                        
                        # Extract facts from chunk
                        try:
                            facts = extract_facts_from_chunk_dspy(
                                chunk_text=chunk.combined_text,
                                previous_context=""
                            )
                            total_facts += len(facts)
                            
                            print(f"   Facts extracted: {len(facts)}")
                            for fact in facts:
                                print(f"      • {fact.get('subject', 'N/A')} {fact.get('predicate', 'N/A')} {fact.get('value', 'N/A')}")
                                
                        except Exception as e:
                            print(f"   ❌ Extraction failed: {e}")
                        
                        # Clear buffer
                        buffer.clear_buffer()
            
            # Process any remaining turns
            if buffer.buffer:
                print(f"\n🔄 FINAL CHUNK (remaining turns):")
                chunk = buffer.force_process()
                if chunk:
                    chunks_processed += 1
                    print(f"   Turns: {chunk.total_turns}")
                    
                    try:
                        facts = extract_facts_from_chunk_dspy(
                            chunk_text=chunk.combined_text,
                            previous_context=""
                        )
                        total_facts += len(facts)
                        
                        print(f"   Facts extracted: {len(facts)}")
                        for fact in facts:
                            print(f"      • {fact.get('subject', 'N/A')} {fact.get('predicate', 'N/A')} {fact.get('value', 'N/A')}")
                    except Exception as e:
                        print(f"   ❌ Extraction failed: {e}")
            
            results.append({
                'conversation': conv_idx,
                'total_turns': len(conversation),
                'chunks_processed': chunks_processed,
                'total_facts': total_facts,
                'facts_per_turn': total_facts / len(conversation) if conversation else 0
            })
            
            print(f"✅ Conversation {conv_idx} complete: {chunks_processed} chunks, {total_facts} facts")
        
        return results
    
    def run_all_tests(self):
        """Run all M3 buffering tests"""
        print("🚀 M3 Conversation Buffering System Test")
        print("=" * 80)
        
        # Test 1: Basic functionality
        self.test_basic_buffer_functionality()
        
        # Test 2: Extraction quality comparison
        extraction_results = self.test_chunk_extraction_quality()
        
        # Test 3: Complete conversation processing
        conversation_results = self.test_conversation_processing()
        
        # Summary
        print("\n🏆 TEST RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"✅ Basic buffer functionality: PASSED")
        print(f"📊 Extraction comparison:")
        print(f"   Chunk advantage: {extraction_results['chunk_advantage']} facts")
        print(f"   Individual: {extraction_results['individual_facts']} facts")
        print(f"   Chunk-based: {extraction_results['chunk_facts']} facts")
        
        print(f"📈 Conversation processing:")
        total_facts = sum(r['total_facts'] for r in conversation_results)
        total_turns = sum(r['total_turns'] for r in conversation_results)
        avg_facts_per_turn = total_facts / total_turns if total_turns else 0
        
        for result in conversation_results:
            print(f"   Conv {result['conversation']}: {result['total_facts']} facts from {result['total_turns']} turns ({result['facts_per_turn']:.2f} facts/turn)")
        
        print(f"\n🎯 Overall: {total_facts} facts from {total_turns} turns ({avg_facts_per_turn:.2f} facts/turn)")
        
        # Assessment
        if extraction_results['chunk_advantage'] > 0:
            print("✅ RESULT: Chunk-based extraction shows improvement!")
        elif extraction_results['chunk_advantage'] == 0:
            print("⚠️ RESULT: Chunk-based extraction equivalent to individual")
        else:
            print("❌ RESULT: Chunk-based extraction needs improvement")
        
        return {
            'extraction_results': extraction_results,
            'conversation_results': conversation_results,
            'total_facts': total_facts,
            'total_turns': total_turns,
            'avg_facts_per_turn': avg_facts_per_turn
        }


def main():
    """Run M3 buffering tests"""
    tester = M3BufferingTester()
    results = tester.run_all_tests()
    return results


if __name__ == "__main__":
    main()