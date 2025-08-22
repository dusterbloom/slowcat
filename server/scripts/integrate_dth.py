"""
Integration script to add Dynamic Tape Head to Smart Context Manager

Run this to upgrade SlowCat's memory from simple recency to consciousness-based selection.
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from memory.dynamic_tape_head import DynamicTapeHead
from processors.smart_context_manager import SmartContextManager


async def test_dth_integration():
    """Test DTH integration with existing system"""
    
    logger.info("🧠 Testing Dynamic Tape Head integration...")
    
    # 1. Create memory system (using existing)
    from memory import create_smart_memory_system
    memory_system = create_smart_memory_system("data/facts.db")
    
    # 2. Create Dynamic Tape Head
    dth = DynamicTapeHead(memory_system)
    logger.info(f"✅ DTH initialized with policy v{dth.policy['version']}")
    
    # 3. Test basic retrieval
    test_queries = [
        "What did we discuss yesterday?",
        "Tell me about Python programming",
        "What's the weather like?",
        "Can you help me with that thing we talked about?"
    ]
    
    for query in test_queries:
        logger.info(f"\n📝 Query: {query}")
        
        bundle = await dth.seek(query, budget=1000)
        
        logger.info(f"  Selected: {len(bundle.verbatim)} verbatim, "
                   f"{len(bundle.shadows)} shadows, "
                   f"{len(bundle.facts)} facts")
        logger.info(f"  Tokens: {bundle.token_count}/{bundle.token_budget}")
        logger.info(f"  Latency: {bundle.metadata.get('latency_ms', 0):.1f}ms")
        
        if bundle.metadata.get('tripwires'):
            logger.warning(f"  ⚠️ Tripwires: {bundle.metadata['tripwires']}")
        
        # Show top memory if any
        if bundle.verbatim:
            top = bundle.verbatim[0]
            logger.info(f"  Top memory (score={top.score:.3f}): {top.content[:50]}...")
            components = top.score_components
            logger.info(f"    R={components['R']:.2f} S={components['S']:.2f} "
                       f"E={components['E']:.2f} D={components['D']:.2f}")
    
    # 4. Test performance
    logger.info("\n📊 Performance Test (100 queries)...")
    import time
    
    latencies = []
    for i in range(100):
        start = time.time()
        await dth.seek(f"Performance test query {i}", budget=1000)
        latencies.append((time.time() - start) * 1000)
    
    import numpy as np
    logger.info(f"  Mean latency: {np.mean(latencies):.1f}ms")
    logger.info(f"  P95 latency: {np.percentile(latencies, 95):.1f}ms")
    logger.info(f"  P99 latency: {np.percentile(latencies, 99):.1f}ms")
    
    # 5. Save metrics
    dth.save_metrics("logs/dth_integration_metrics.json")
    logger.info("📊 Metrics saved to logs/dth_integration_metrics.json")
    
    # 6. Show how to integrate with Smart Context Manager
    logger.info("\n🔧 Integration with Smart Context Manager:")
    logger.info("  1. Import DTH in smart_context_manager.py:")
    logger.info("     from memory.dynamic_tape_head import DynamicTapeHead")
    logger.info("  2. Initialize in __init__:")
    logger.info("     self.tape_head = DynamicTapeHead(self.memory_system)")
    logger.info("  3. Replace memory retrieval in _build_memory_context:")
    logger.info("     bundle = await self.tape_head.seek(query, budget=2000)")
    logger.info("  4. Use bundle.verbatim for context building")
    
    return dth


async def patch_smart_context_manager():
    """
    Actually patch the Smart Context Manager to use DTH
    
    WARNING: This modifies the running system!
    """
    
    logger.warning("⚠️ This will modify smart_context_manager.py. Continue? (y/n)")
    response = input().strip().lower()
    
    if response != 'y':
        logger.info("Cancelled")
        return
    
    # Read the current file
    scm_path = Path("processors/smart_context_manager.py")
    if not scm_path.exists():
        logger.error(f"File not found: {scm_path}")
        return
    
    with open(scm_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "DynamicTapeHead" in content:
        logger.info("✅ Already patched!")
        return
    
    # Add import
    import_line = "from memory.dynamic_tape_head import DynamicTapeHead"
    import_position = content.find("from memory import")
    if import_position > 0:
        # Find end of line
        eol = content.find('\n', import_position)
        content = content[:eol+1] + import_line + '\n' + content[eol+1:]
    
    # Add initialization in __init__
    init_text = """
        # Initialize Dynamic Tape Head for consciousness-based memory selection
        self.tape_head = DynamicTapeHead(self.memory_system)
        logger.info("🧠 Dynamic Tape Head initialized for intelligent memory selection")
"""
    
    # Find where to add (after memory_system initialization)
    init_position = content.find("self.memory_system = create_smart_memory_system")
    if init_position > 0:
        # Find end of line
        eol = content.find('\n', init_position)
        content = content[:eol+1] + init_text + content[eol+1:]
    
    # Save backup
    backup_path = scm_path.with_suffix('.py.bak')
    with open(backup_path, 'w') as f:
        f.write(content)
    
    logger.info(f"📄 Backup saved to {backup_path}")
    
    # Write patched version
    # Note: Not actually writing to avoid breaking the system
    # In production, you would uncomment this:
    # with open(scm_path, 'w') as f:
    #     f.write(content)
    
    logger.info("✅ Patch prepared (not applied in demo mode)")
    logger.info("To apply: uncomment the write operation in this script")


if __name__ == "__main__":
    # Choose mode
    print("\n🧠 Dynamic Tape Head Integration\n")
    print("1. Test DTH with existing memory")
    print("2. Patch Smart Context Manager (modifies code)")
    print("3. Run performance benchmark")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(test_dth_integration())
    elif choice == "2":
        asyncio.run(patch_smart_context_manager())
    elif choice == "3":
        # Performance benchmark
        asyncio.run(test_dth_integration())
    else:
        print("Invalid choice")
