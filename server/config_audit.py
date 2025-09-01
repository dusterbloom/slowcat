#!/usr/bin/env python3
"""
Configuration Audit for Slowcat
Analyze all configuration options and categorize them
"""

import re
import os
from pathlib import Path

def audit_env_file():
    """Audit .env file for configuration options"""
    env_path = Path(__file__).parent / ".env"
    
    options = []
    with open(env_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and line.startswith(('#', '//')):
                continue
            
            match = re.match(r'^([A-Z_][A-Z0-9_]*)=(.*)$', line)
            if match:
                var_name = match.group(1)
                value = match.group(2)
                options.append({
                    'name': var_name,
                    'value': value,
                    'line': line_num,
                    'source': '.env'
                })
    
    return options

def categorize_options(options):
    """Categorize options as essential, nice-to-have, or internal"""
    
    categories = {
        'essential': [],
        'nice_to_have': [],
        'internal': []
    }
    
    # Essential: Core functionality users MUST configure
    essential_patterns = [
        r'^OPENAI_(API_KEY|BASE_URL)$',
        r'^LLM_.*',
        r'^STT_BACKEND$',
        r'^TTS_ENGINE$',
        r'^ENABLE_(MEMORY|VOICE_RECOGNITION|MCP)$',
        r'^SURREALDB_(URL|USER|PASS|NAMESPACE|DATABASE)$',
        r'^USER_ID$',
        r'^ASSISTANT_ID$'
    ]
    
    # Nice-to-have: Features users might want to configure
    nice_to_have_patterns = [
        r'^ENABLE_(VIDEO|REFLECTIONS|THINKING)$',
        r'^SHERPA_.*LANGUAGE.*',
        r'^REFLECTION_.*',
        r'^MCPO_.*',
        r'^PIPELINE_IDLE_TIMEOUT.*',
        r'^SC_.*BUDGET.*',
        r'^DSPY_.*',
        r'^DISABLE_ALL_TOOLS$'
    ]
    
    # Internal: Technical settings that should be constants
    internal_patterns = [
        r'^MLX_.*',
        r'^METAL_.*',
        r'^MTL_.*',
        r'^HF_HUB_.*',
        r'^TRANSFORMERS_.*',
        r'^OBJC_.*',
        r'^SHERPA_.*CHUNK.*',
        r'^SHERPA_.*THREADS.*',
        r'^SHERPA_.*PATHS.*',
        r'^SHERPA_.*PARTIAL.*',
        r'^.*VERBOSE$',
        r'^.*LOG_.*',
        r'^.*DEBUG.*',
        r'^ROUTER_THRESHOLD.*',
        r'^.*COOLDOWN.*',
        r'^.*DEBOUNCE.*',
        r'^.*MIN_.*MS$',
        r'^.*_TOKENS$',
        r'^CONTEXT7_API_KEY$',
        r'^DEEPGRAM_API_KEY$',
        r'^RIME_API_KEY$'
    ]
    
    for option in options:
        name = option['name']
        category = 'internal'  # default
        
        # Check essential first
        for pattern in essential_patterns:
            if re.match(pattern, name):
                category = 'essential'
                break
        
        # Check nice-to-have if not essential
        if category != 'essential':
            for pattern in nice_to_have_patterns:
                if re.match(pattern, name):
                    category = 'nice_to_have'
                    break
        
        categories[category].append(option)
    
    return categories

def main():
    """Audit configuration complexity"""
    print("📊 CONFIGURATION COMPLEXITY AUDIT")
    print("=" * 50)
    
    # Audit .env file
    env_options = audit_env_file()
    print(f"Total .env options: {len(env_options)}")
    
    # Categorize options
    categories = categorize_options(env_options)
    
    print(f"\n📋 CATEGORIZATION RESULTS:")
    print(f"Essential (must configure): {len(categories['essential'])}")
    print(f"Nice-to-have (optional): {len(categories['nice_to_have'])}")
    print(f"Internal (should be constants): {len(categories['internal'])}")
    
    print(f"\n✅ ESSENTIAL OPTIONS ({len(categories['essential'])}):")
    for opt in sorted(categories['essential'], key=lambda x: x['name']):
        print(f"  {opt['name']}={opt['value']}")
    
    print(f"\n🔧 NICE-TO-HAVE OPTIONS ({len(categories['nice_to_have'])}):")
    for opt in sorted(categories['nice_to_have'], key=lambda x: x['name']):
        print(f"  {opt['name']}={opt['value']}")
    
    print(f"\n⚙️  INTERNAL OPTIONS (candidates for code constants) ({len(categories['internal'])}):")
    for opt in sorted(categories['internal'], key=lambda x: x['name']):
        print(f"  {opt['name']}={opt['value']}")
    
    # Recommended essential options
    essential_count = len(categories['essential'])
    nice_count = len(categories['nice_to_have']) 
    total_recommended = essential_count + min(nice_count, 10)  # Cap nice-to-have
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"  Essential options: {essential_count}")
    print(f"  Selected nice-to-have: {min(nice_count, 10)}")
    print(f"  Total user-configurable: {total_recommended}")
    print(f"  Move to constants: {len(categories['internal'])}")
    print(f"  Complexity reduction: {len(env_options)} → {total_recommended} options")
    
    # Save detailed results
    with open('config_audit_results.txt', 'w') as f:
        f.write("CONFIGURATION AUDIT RESULTS\n")
        f.write("=" * 30 + "\n\n")
        
        f.write(f"ESSENTIAL OPTIONS ({len(categories['essential'])}):\n")
        for opt in sorted(categories['essential'], key=lambda x: x['name']):
            f.write(f"{opt['name']}={opt['value']}\n")
        
        f.write(f"\nNICE-TO-HAVE OPTIONS ({len(categories['nice_to_have'])}):\n")
        for opt in sorted(categories['nice_to_have'], key=lambda x: x['name']):
            f.write(f"{opt['name']}={opt['value']}\n")
        
        f.write(f"\nINTERNAL OPTIONS ({len(categories['internal'])}):\n")
        for opt in sorted(categories['internal'], key=lambda x: x['name']):
            f.write(f"{opt['name']}={opt['value']}\n")
    
    print(f"\n📄 Detailed results saved to: config_audit_results.txt")
    
    return categories

if __name__ == "__main__":
    main()