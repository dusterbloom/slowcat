"""
Builds the system prompt for the LLM with simplified tool documentation
"""

from typing import List, Dict
from pipecat.adapters.schemas.function_schema import FunctionSchema
import string

class SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, f"{{{key}}}")
        else:
            return super().get_value(key, args, kwargs)

def safe_format(prompt: str, **kwargs) -> str:
    """Safely formats a string, ignoring missing keys."""
    formatter = SafeFormatter()
    return formatter.format(prompt, **kwargs)

def _categorize_tools(tools: List[FunctionSchema]) -> Dict[str, List[str]]:
    """Categorize tools by their domain for clearer documentation"""
    categories = {
        "memory": [],
        "filesystem": [],
        "search": [],
        "browser": [],
        "calculate": [],
        "other": []
    }
    
    for tool in tools:
        name = tool.name.lower()
        if "memory" in name:
            categories["memory"].append(tool.name)
        elif "filesystem" in name or "file" in name:
            categories["filesystem"].append(tool.name)
        elif "search" in name or "brave" in name:
            categories["search"].append(tool.name)
        elif "browser" in name or "javascript" in name:
            categories["browser"].append(tool.name)
        elif "calculate" in name:
            categories["calculate"].append(tool.name)
        else:
            categories["other"].append(tool.name)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def _generate_simple_tool_docs(local_tools: List[FunctionSchema], mcp_tools: List[FunctionSchema]) -> str:
    """Generate simplified tool documentation with examples"""
    
    all_tools = local_tools + mcp_tools
    if not all_tools:
        return ""
    
    # Categorize tools
    categories = _categorize_tools(all_tools)
    
    # Build documentation
    docs = "\n## Available Tools\n\n"
    docs += "You have direct access to these tools. Call them by name with appropriate parameters:\n\n"
    
    # Add categorized tool list
    for category, tool_names in categories.items():
        docs += f"**{category.title()}:**\n"
        for tool_name in tool_names[:5]:  # Limit to first 5 per category
            docs += f"- {tool_name}\n"
        if len(tool_names) > 5:
            docs += f"- ... and {len(tool_names) - 5} more\n"
        docs += "\n"
    
    # Add simple examples
    docs += "**Examples:**\n"
    docs += "- Search memory: `memory_search_nodes(query='favorite number')`\n"
    docs += "- Read file: `filesystem_read_file(path='~/Desktop/notes.txt')`\n"
    docs += "- Web search: `brave_web_search(query='latest news')`\n"
    docs += "- Calculate: `calculate(expression='2 + 2')`\n"
    
    return docs

def generate_final_system_prompt(base_prompt: str, local_tools: List[FunctionSchema], mcp_tools: List[FunctionSchema]) -> str:
    """
    Generates the full system prompt with simplified tool documentation
    """
    
    # Generate simplified tool documentation
    tool_docs = _generate_simple_tool_docs(local_tools, mcp_tools)
    
    # Inject the tool documentation into the placeholder
    final_prompt = safe_format(base_prompt, tool_definitions_placeholder=tool_docs)
    
    return final_prompt