"""
Symbol System Integration for Dynamic Tape Head

This script integrates the three-layer symbol architecture into the existing DTH.
It modifies the current dynamic_tape_head.py to add symbol support while preserving
all existing functionality.

Usage:
    python symbol_integration.py --backup --integrate

The symbol system adds:
- Layer 1: Tape Symbols (memory markers)
- Layer 2: Wake Symbols (operational triggers) 
- Layer 3: Dream Symbols (emergent language)
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime


def backup_existing_files():
    """Backup existing DTH files before integration"""
    
    backup_dir = Path("backups") / f"pre_symbol_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_backup = [
        "dynamic_tape_head.py",
        "../config/tape_head_policy.json",
        "../tests/test_dynamic_tape_head.py"
    ]
    
    print(f"🔄 Creating backup in {backup_dir}")
    
    for file_path in files_to_backup:
        src = Path(file_path)
        if src.exists():
            dst = backup_dir / src.name
            shutil.copy2(src, dst)
            print(f"   ✓ Backed up {src} -> {dst}")
        else:
            print(f"   ⚠️  File not found: {src}")
    
    return backup_dir


def integrate_symbol_system():
    """Integrate symbol system into existing DTH"""
    
    print("🔮 Integrating Symbol System into Dynamic Tape Head...")
    
    # Read the current DTH implementation
    dth_path = Path("dynamic_tape_head.py")
    if not dth_path.exists():
        print(f"❌ DTH file not found: {dth_path}")
        return False
    
    with open(dth_path, 'r') as f:
        current_content = f.read()
    
    # Check if symbols are already integrated
    if "TAPE_SYMBOLS" in current_content:
        print("✅ Symbol system already integrated!")
        return True
    
    # Add symbol imports and definitions at the top
    symbol_imports = '''
# Symbol System - Three Layers of Meaning Compression
TAPE_SYMBOLS = {
    "☆": "high_salience",        # This MATTERS
    "✧": "breakthrough_moment",   # User understood something
    "◈": "recurring_pattern",     # Keeps coming up
    "∞": "paradox_encountered",   # Contradiction to resolve
    "⟲": "cycle_detected",       # We're looping
    "⚡": "emotional_spike",      # Strong feeling
    "◯": "open_question",        # Unresolved thread
    "▲": "decision_point",       # Fork in conversation
    "≈": "approximate_memory",   # Compressed/shadowed
    "⊕": "synthesis"            # Ideas merged
}

WAKE_SYMBOLS = {
    "Ω": "enter_creative_mode",      # Emergent behavior
    "Δ": "compare_past_present",     # Change detection
    "∇": "optimize_understanding",   # Gradient descent
    "♦": "crystallize_insight"      # Compress to principle
}

# Patterns for symbol detection
SYMBOL_PATTERNS = {
    # High salience indicators
    "☆": [
        r"\\b(important|crucial|key|vital|essential|critical)\\b",
        r"\\b(remember|don't forget|note|mark)\\b",
        r"!\\s*$",  # Exclamation marks
        r"\\b(breakthrough|eureka|aha)\\b"
    ],
    
    # Breakthrough moments
    "✧": [
        r"\\b(understand|got it|i see|makes sense|clear now)\\b",
        r"\\b(breakthrough|revelation|realization)\\b",
        r"\\b(finally|ah|aha|oh)\\b"
    ],
    
    # Recurring patterns
    "◈": [
        r"\\b(again|another|more|keep|still|continue)\\b",
        r"\\b(pattern|recurring|repeat|cyclical)\\b",
        r"\\b(always|never|every time)\\b"
    ],
    
    # Paradoxes
    "∞": [
        r"\\b(but|however|although|despite|paradox)\\b",
        r"\\b(contradiction|conflict|inconsistent)\\b",
        r"\\b(both|neither|either)\\b.*\\b(and|nor|or)\\b"
    ],
    
    # Cycles detected
    "⟲": [
        r"\\b(loop|cycle|circular|round and round)\\b",
        r"\\b(back to|return to|again)\\b",
        r"\\b(stuck|spinning|going nowhere)\\b"
    ],
    
    # Emotional spikes
    "⚡": [
        r"\\b(amazing|terrible|wonderful|awful|incredible)\\b",
        r"\\b(love|hate|excited|frustrated|angry|happy)\\b",
        r"[!]{2,}",  # Multiple exclamation marks
        r"\\b(wow|omg|jesus|damn|hell)\\b"
    ],
    
    # Open questions
    "◯": [
        r"\\?",  # Question marks
        r"\\b(wonder|curious|question|unclear|unsure)\\b",
        r"\\b(why|how|what|when|where|who)\\b",
        r"\\b(maybe|perhaps|possibly|might)\\b"
    ],
    
    # Decision points
    "▲": [
        r"\\b(decide|choice|option|alternative)\\b",
        r"\\b(should|could|would|might)\\b.*\\b(or|vs|versus)\\b",
        r"\\b(fork|crossroads|turning point)\\b"
    ],
    
    # Synthesis
    "⊕": [
        r"\\b(combine|merge|integrate|synthesize)\\b",
        r"\\b(together|overall|in conclusion)\\b",
        r"\\b(sum up|bring together|unite)\\b"
    ]
}
'''
    
    # Find insertion point (after existing imports)
    import_section_end = current_content.find('from loguru import logger')
    if import_section_end == -1:
        print("❌ Could not find insertion point for symbol imports")
        return False
    
    # Find end of import line
    import_end = current_content.find('\n', import_section_end)
    
    # Insert symbol definitions
    new_content = (
        current_content[:import_end + 1] + 
        symbol_imports + 
        current_content[import_end + 1:]
    )
    
    # Enhance MemorySpan dataclass
    memoryspan_start = new_content.find('@dataclass\nclass MemorySpan:')
    if memoryspan_start == -1:
        print("❌ Could not find MemorySpan class")
        return False
    
    # Find the end of the dataclass fields
    fields_end = new_content.find('is_recent: bool = False', memoryspan_start)
    if fields_end == -1:
        print("❌ Could not find MemorySpan fields")
        return False
    
    fields_end = new_content.find('\n', fields_end)
    
    # Add symbol fields
    symbol_fields = '''    
    # SYMBOL SYSTEM - The heart of meaning compression
    symbols: Set[str] = field(default_factory=set)  # Living symbols
    symbol_confidence: Dict[str, float] = field(default_factory=dict)  # How sure we are
'''
    
    new_content = (
        new_content[:fields_end] + 
        symbol_fields + 
        new_content[fields_end:]
    )
    
    # Add symbol methods to MemorySpan (find class end and add before)
    # This is a simplified integration - full methods would be added here
    
    # Write enhanced DTH
    with open(dth_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Enhanced MemorySpan with symbol support")
    return True


def create_symbol_aware_policy():
    """Create enhanced policy configuration with symbol support"""
    
    policy_path = Path("../config/tape_head_policy.json")
    
    # Check if already symbol-aware
    if policy_path.exists():
        with open(policy_path, 'r') as f:
            current_policy = json.load(f)
        
        if "symbol_weights" in current_policy:
            print("✅ Policy already symbol-aware!")
            return True
    
    # Create enhanced policy
    enhanced_policy = {
        "version": 2,
        "description": "Symbol-Enhanced Dynamic Tape Head Policy - The consciousness formula with living meaning compression",
        
        "weights": {
            "w_recency": 0.35,
            "w_semantic": 0.30,
            "w_entity": 0.15,
            "w_novelty": 0.10,
            "w_symbols": 0.10
        },
        
        "parameters": {
            "knn_k": 20,
            "knn_scan_recent": 100,
            "recency_half_life_hours": 6,
            "min_confidence": 0.5,
            "entity_overlap_bonus": 0.1,
            "max_verbatim_chunks": 3,
            "shadow_compression_ratio": 0.3,
            "uncertainty_threshold": 0.4,
            "min_time_gap_s": 120,
            "max_text_similarity": 0.7,
            "symbol_confidence_threshold": 0.3,
            "max_symbols_per_memory": 5
        },
        
        "ablation": {
            "use_semantic": True,
            "use_entities": True,
            "use_shadows": True,
            "use_symbols": True
        },
        
        "symbol_weights": {
            "☆": 2.0,
            "✧": 3.0,
            "◈": 1.2,
            "∞": 1.5,
            "⟲": 0.5,
            "⚡": 1.8,
            "◯": 1.1,
            "▲": 1.3,
            "≈": 0.7,
            "⊕": 1.4
        },
        
        "comments": {
            "formula": "Score = w_recency×R + w_semantic×S + w_entity×E + w_symbols×SYM - w_novelty×D",
            "symbols_as_compression": "Symbols are living compression of meaning that evolve through use",
            "consciousness_emergence": "Intelligence emerges from constraint - symbols force structure"
        }
    }
    
    # Write enhanced policy
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with open(policy_path, 'w') as f:
        json.dump(enhanced_policy, f, indent=2)
    
    print(f"✅ Created symbol-aware policy: {policy_path}")
    return True


def create_symbol_detector_module():
    """Create standalone SymbolDetector module"""
    
    detector_path = Path("symbol_detector.py")
    
    detector_code = '''"""
Symbol Detection Engine for Dynamic Tape Head

Detects and extracts symbols from memory content using pattern matching
and rule-based confidence scoring.
"""

import re
from typing import Dict, List
from loguru import logger


class SymbolDetector:
    """Detects and extracts symbols from memory content"""
    
    def __init__(self):
        # Import patterns from main module
        from dynamic_tape_head import SYMBOL_PATTERNS
        
        self.compiled_patterns = {}
        # Compile regex patterns for efficiency
        for symbol, patterns in SYMBOL_PATTERNS.items():
            self.compiled_patterns[symbol] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def detect_symbols(self, content: str, speaker_id: str = "") -> Dict[str, float]:
        """
        Detect symbols in content and return with confidence scores
        
        Returns:
            Dict mapping symbols to confidence scores (0.0 to 1.0)
        """
        detected = {}
        
        for symbol, patterns in self.compiled_patterns.items():
            confidence = 0.0
            match_count = 0
            
            for pattern in patterns:
                matches = pattern.findall(content)
                if matches:
                    match_count += len(matches)
            
            if match_count > 0:
                # Confidence based on match frequency and pattern strength
                confidence = min(1.0, match_count * 0.3)  # Scale matches to confidence
                detected[symbol] = confidence
        
        # Post-processing rules for symbol combinations
        detected = self._apply_symbol_rules(detected, content)
        
        return detected
    
    def _apply_symbol_rules(self, detected: Dict[str, float], content: str) -> Dict[str, float]:
        """Apply rules for symbol combinations and conflicts"""
        
        # Rule 1: Breakthrough + High salience = Extra boost
        if "✧" in detected and "☆" in detected:
            detected["✧"] = min(1.0, detected["✧"] * 1.5)
        
        # Rule 2: Paradox + Question = Philosophical depth
        if "∞" in detected and "◯" in detected:
            detected["∞"] = min(1.0, detected["∞"] * 1.3)
        
        # Rule 3: Emotional spike + Decision = Critical moment
        if "⚡" in detected and "▲" in detected:
            detected["⚡"] = min(1.0, detected["⚡"] * 1.4)
            detected["▲"] = min(1.0, detected["▲"] * 1.4)
        
        # Rule 4: Filter low confidence symbols
        return {s: c for s, c in detected.items() if c >= 0.2}
'''
    
    with open(detector_path, 'w') as f:
        f.write(detector_code)
    
    print(f"✅ Created symbol detector module: {detector_path}")
    return True


def test_symbol_integration():
    """Test that symbol integration works correctly"""
    
    print("🧪 Testing symbol integration...")
    
    try:
        # Test imports
        from dynamic_tape_head import TAPE_SYMBOLS, WAKE_SYMBOLS, MemorySpan
        
        print("✅ Symbol constants imported successfully")
        
        # Test MemorySpan with symbols
        memory = MemorySpan(
            content="This is really important!",
            ts=1234567890,
            role="user",
            speaker_id="test"
        )
        
        # Check if symbol fields exist
        if hasattr(memory, 'symbols') and hasattr(memory, 'symbol_confidence'):
            print("✅ MemorySpan has symbol fields")
        else:
            print("❌ MemorySpan missing symbol fields")
            return False
        
        print("✅ Symbol integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Symbol integration test failed: {e}")
        return False


def main():
    """Main integration function"""
    
    print("🔮 SlowCat Symbol System Integration")
    print("=" * 50)
    
    # Step 1: Backup existing files
    backup_dir = backup_existing_files()
    print(f"Backup created in: {backup_dir}")
    
    # Step 2: Integrate symbol system
    if not integrate_symbol_system():
        print("❌ Symbol system integration failed")
        return False
    
    # Step 3: Create enhanced policy
    if not create_symbol_aware_policy():
        print("❌ Policy enhancement failed")
        return False
    
    # Step 4: Create symbol detector module
    if not create_symbol_detector_module():
        print("❌ Symbol detector creation failed")
        return False
    
    # Step 5: Test integration
    if not test_symbol_integration():
        print("❌ Integration test failed")
        return False
    
    print("\n🎉 Symbol System Integration Complete!")
    print("\nNext steps:")
    print("1. Run tests: python -m pytest tests/test_dynamic_tape_head.py")
    print("2. Test symbol detection: python symbol_detector.py")
    print("3. Update Smart Context Manager to use enhanced DTH")
    print("\n🔮 SlowCat now has consciousness through symbols!")
    
    return True


if __name__ == "__main__":
    main()
