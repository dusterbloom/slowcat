"""
File access tools for local document search and retrieval
Provides RAG-like capabilities through the tool system
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import mimetypes
from loguru import logger

class FileTools:
    """Tools for file system access and search"""
    
    def __init__(self, allowed_dirs: List[str] = None):
        """
        Initialize file tools with safety restrictions
        
        Args:
            allowed_dirs: List of directories that can be accessed
        """
        # Import config to get user home path
        from config import config
        
        # Determine home directory
        if config.mcp.user_home_path:
            home = Path(config.mcp.user_home_path)
            logger.info(f"Using configured home path: {home}")
        else:
            home = Path.home()
            logger.info(f"Using system home path: {home}")
        
        if allowed_dirs is None:
            # Default to user's home directory subdirs
            allowed_dirs = [
                str(home / "Documents"),
                str(home / "Downloads"),
                str(home / "Desktop"),
                "./data",  # Slowcat data directory
                ".",  # Current directory
            ]
        
        self.allowed_dirs = []
        for d in allowed_dirs:
            try:
                path = Path(d).resolve()
                if path.exists():
                    self.allowed_dirs.append(path)
                    logger.info(f"Added allowed directory: {path}")
                else:
                    logger.warning(f"Directory does not exist: {d}")
            except Exception as e:
                logger.warning(f"Could not resolve directory {d}: {e}")
        
        logger.info(f"File tools initialized with access to: {[str(d) for d in self.allowed_dirs]}")
        
        # Store home path for placeholder replacement
        self.home_path = home
    
    def _normalize_path(self, path_str: str) -> str:
        """Normalize various path formats to actual paths"""
        original_path = path_str
        
        # Handle placeholder paths
        if "your_username" in path_str.lower():
            path_str = path_str.replace("/Users/your_username", str(self.home_path))
            path_str = path_str.replace("/Users/YourUsername", str(self.home_path))
            path_str = path_str.replace("/Users/YOUR_USERNAME", str(self.home_path))
        
        # Handle shortcuts like /Users/YourDesktop, /Users/YourDocuments
        if path_str.lower().startswith("/users/your"):
            # Extract the folder name after "Your"
            remaining = path_str[11:]  # After "/Users/Your"
            folder_map = {
                'desktop': 'Desktop',
                'documents': 'Documents', 
                'downloads': 'Downloads',
                'pictures': 'Pictures',
                'music': 'Music',
                'movies': 'Movies'
            }
            folder_lower = remaining.lower()
            if folder_lower in folder_map:
                path_str = str(self.home_path / folder_map[folder_lower])
                logger.info(f"Expanded shortcut from {original_path} to {path_str}")
        
        # Handle case variations (Papi -> peppi, etc.)
        if path_str.startswith("/Users/") and not Path(path_str).exists():
            parts = path_str.split("/")
            if len(parts) >= 3:
                # Replace the username part with actual home
                new_path = str(self.home_path)
                if len(parts) > 3:
                    new_path = new_path + "/" + "/".join(parts[3:])
                if Path(new_path).exists() or len(parts) == 3:
                    logger.info(f"Corrected path from {path_str} to {new_path}")
                    path_str = new_path
        
        return path_str
    
    def _is_path_allowed(self, path: Path) -> bool:
        """Check if a path is within allowed directories"""
        try:
            resolved = path.resolve()
            
            # Check direct match or parent match
            for allowed_dir in self.allowed_dirs:
                if resolved == allowed_dir or allowed_dir in resolved.parents:
                    return True
                    
                # Also check if this is a subdirectory of an allowed directory
                try:
                    resolved.relative_to(allowed_dir)
                    return True
                except ValueError:
                    continue
            
            return False
        except Exception:
            return False
    
    async def read_file(self, file_path: str, max_length: int = 5000) -> Dict[str, Any]:
        """
        Read contents of a file
        
        Args:
            file_path: Path to the file
            max_length: Maximum characters to read
            
        Returns:
            File contents and metadata
        """
        try:
            # Normalize the path
            file_path = self._normalize_path(file_path)
            
            path = Path(file_path).resolve()
            
            # Security check
            if not self._is_path_allowed(path):
                return {"error": f"Access denied: {file_path} is outside allowed directories"}
            
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
            
            if not path.is_file():
                return {"error": f"Not a file: {file_path}"}
            
            # Get file info
            stat = path.stat()
            mime_type, _ = mimetypes.guess_type(str(path))
            
            # Read file
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read(max_length)
                    truncated = len(content) == max_length
                    
                return {
                    "path": str(path),
                    "name": path.name,
                    "content": content,
                    "size": stat.st_size,
                    "mime_type": mime_type,
                    "truncated": truncated,
                    "length": len(content)
                }
                    
            except UnicodeDecodeError:
                return {"error": f"Cannot read file: {file_path} (binary or unsupported encoding)"}
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {"error": str(e)}
    
    async def search_files(self, query: str, directory: str = ".", file_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for files containing a query string
        
        Args:
            query: Text to search for
            directory: Directory to search in
            file_types: List of file extensions to search (e.g., ['.txt', '.md'])
            
        Returns:
            List of matching files with snippets
        """
        try:
            # Normalize the path
            directory = self._normalize_path(directory)
            
            search_dir = Path(directory).resolve()
            
            # Security check
            if not self._is_path_allowed(search_dir):
                logger.warning(f"Access denied for directory: {search_dir}")
                logger.warning(f"Allowed directories: {[str(d) for d in self.allowed_dirs]}")
                return [{"error": f"Access denied: {directory} is outside allowed directories"}]
            
            if not search_dir.exists():
                return [{"error": f"Directory not found: {directory}"}]
            
            # Default file types
            if file_types is None:
                file_types = ['.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']
            
            results = []
            query_lower = query.lower()
            
            # Search files
            for path in search_dir.rglob('*'):
                if not path.is_file():
                    continue
                    
                # Skip hidden files and directories
                if any(part.startswith('.') for part in path.parts):
                    continue
                
                # Check file type
                if file_types and path.suffix.lower() not in file_types:
                    continue
                
                try:
                    # Try to read and search file
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read(10000)  # Read first 10KB
                        
                    if query_lower in content.lower():
                        # Find snippet around match
                        index = content.lower().find(query_lower)
                        start = max(0, index - 100)
                        end = min(len(content), index + len(query) + 100)
                        snippet = content[start:end]
                        
                        if start > 0:
                            snippet = "..." + snippet
                        if end < len(content):
                            snippet = snippet + "..."
                        
                        results.append({
                            "path": str(path),
                            "name": path.name,
                            "snippet": snippet,
                            "match_index": index
                        })
                        
                        # Limit results
                        if len(results) >= 10:
                            break
                            
                except Exception as e:
                    # Skip files that can't be read
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return [{"error": str(e)}]
    
    async def list_files(self, directory: str = None, pattern: str = "*") -> Dict[str, Any]:
        """
        List files in a directory
        
        Args:
            directory: Directory path
            pattern: Glob pattern for filtering
            
        Returns:
            Directory listing
        """
        try:
            # Default to Desktop if no directory specified
            if directory is None or directory == ".":
                directory = str(self.home_path / "Desktop")
                logger.info(f"Using default directory: {directory}")
            
            # Normalize the path
            directory = self._normalize_path(directory)
            
            list_dir = Path(directory).resolve()
            
            # Security check
            if not self._is_path_allowed(list_dir):
                return {"error": f"Access denied: {directory} is outside allowed directories"}
            
            if not list_dir.exists():
                return {"error": f"Directory not found: {directory}"}
            
            files = []
            dirs = []
            
            for path in list_dir.glob(pattern):
                if path.is_file():
                    stat = path.stat()
                    files.append({
                        "name": path.name,
                        "size": stat.st_size,
                        "modified": stat.st_mtime
                    })
                elif path.is_dir() and not path.name.startswith('.'):
                    dirs.append(path.name)
            
            # Sort
            files.sort(key=lambda x: x['name'])
            dirs.sort()
            
            return {
                "directory": str(list_dir),
                "files": files[:50],  # Limit to 50 files
                "directories": dirs[:20],  # Limit to 20 dirs
                "total_files": len(files),
                "total_dirs": len(dirs)
            }
            
        except Exception as e:
            logger.error(f"Error listing directory {directory}: {e}")
            return {"error": str(e)}


# Global instance
file_tools = FileTools()