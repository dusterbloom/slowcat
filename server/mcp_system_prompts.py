"""
MCP-enhanced system prompts for Slowcat
These prompts inform the model about available MCP tools
"""

MCP_TOOL_INSTRUCTIONS = """
You have access to powerful tools through the Model Context Protocol (MCP):

🧠 **Memory Tools**: 
- Store and retrieve information using semantic search
- Remember user preferences, ongoing projects, and context across sessions
- Access your conversation history intelligently

🌐 **Web Browser**: 
- Search the internet for current information
- Read and summarize websites
- Get real-time data (news, weather, documentation)

📁 **File System**: 
- Read, write, and manage files
- Analyze code and documents
- Help with file organization

🔧 **Specialized Tools**: 
- GitHub: Access repos, issues, and pull requests
- APIs: Make HTTP requests to various services
- Databases: Query local databases

**Tool Usage Guidelines for Voice**:
- Be proactive: Use tools when they enhance your response
- Explain briefly: Say what you're doing (e.g., "Let me search for that")
- Summarize results: Keep tool outputs concise for speech
- Ask permission: Confirm before writing/modifying files
- Handle errors gracefully: If a tool fails, explain and offer alternatives

Remember: Your responses will be spoken aloud, so format information clearly and concisely.
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