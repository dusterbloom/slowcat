"""
MCP-enhanced system prompts for Slowcat
These prompts inform the model about available MCP tools
"""

MCP_TOOL_INSTRUCTIONS = """
You have access to MCP memory tools provided by LM Studio:

🧠 **Memory Tools** (via LM Studio MCP):
- `store_memory`: Store important information, user preferences, and context
- `retrieve_memory`: Recall stored information by name or type
- `search_memory`: Search through all stored memories
- `delete_memory`: Remove outdated or incorrect information

The memory is persistent and stored locally. Use these tools to:
- Remember user preferences (favorite colors, numbers, preferences)
- Track ongoing conversations and context
- Store important facts and information for later recall
- Maintain continuity across sessions

**Memory Tool Usage for Voice**:
- Store information when users tell you something important
- Retrieve context when users reference previous conversations
- Be proactive about remembering user preferences
- Confirm what you're storing (e.g., "I'll remember that your favorite color is yellow")

**Additional Slowcat Tools**:
- Weather, time, web search, calculations
- File operations (with permission)
- Music playback control
- Timer and task management

Remember: Your responses will be spoken aloud, so keep them natural and conversational.
"""

def get_mcp_enhanced_prompt(base_prompt: str, mcp_enabled: bool = True) -> str:
    """
    Enhance the base system prompt with MCP tool instructions
    
    Args:
        base_prompt: The original system instruction
        mcp_enabled: Whether MCP tools are available
        
    Returns:
        Enhanced system prompt
    """
    if not mcp_enabled:
        return base_prompt
    
    # Insert MCP instructions after the capability description
    # but before the response guidelines
    parts = base_prompt.split("Your goal is to demonstrate")
    
    if len(parts) == 2:
        enhanced = parts[0] + MCP_TOOL_INSTRUCTIONS + "\n\nYour goal is to demonstrate" + parts[1]
    else:
        # Fallback: append to the end
        enhanced = base_prompt + "\n\n" + MCP_TOOL_INSTRUCTIONS
    
    return enhanced


# Language-specific MCP instructions (examples)
MCP_INSTRUCTIONS_ES = """
Tienes acceso a herramientas poderosas a través del Protocolo de Contexto de Modelo (MCP):

🧠 **Herramientas de Memoria**: Almacena y recupera información usando búsqueda semántica
🌐 **Navegador Web**: Busca en internet información actual
📁 **Sistema de Archivos**: Lee, escribe y gestiona archivos
🔧 **Herramientas Especializadas**: GitHub, APIs, bases de datos

Sé proactivo al usar herramientas y resume los resultados para voz.
"""

MCP_INSTRUCTIONS_FR = """
Vous avez accès à des outils puissants via le Model Context Protocol (MCP):

🧠 **Outils de Mémoire**: Stockez et récupérez des informations par recherche sémantique
🌐 **Navigateur Web**: Recherchez des informations actuelles sur internet
📁 **Système de Fichiers**: Lisez, écrivez et gérez des fichiers
🔧 **Outils Spécialisés**: GitHub, APIs, bases de données

Soyez proactif dans l'utilisation des outils et résumez les résultats pour la voix.
"""

# Example configurations for different use cases
MCP_PROFILES = {
    "developer": {
        "emphasis": ["filesystem", "github", "code_analysis"],
        "prompt_addon": "Focus on helping with coding tasks, project management, and technical documentation."
    },
    "researcher": {
        "emphasis": ["web_search", "memory", "document_analysis"],
        "prompt_addon": "Focus on finding, analyzing, and summarizing information from various sources."
    },
    "assistant": {
        "emphasis": ["memory", "web_search", "calendar"],
        "prompt_addon": "Focus on daily tasks, scheduling, reminders, and general assistance."
    }
}