"""
Music library scanner for local audio files
Scans directories for audio files and extracts metadata
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import hashlib
from loguru import logger

# Audio file extensions to scan
AUDIO_EXTENSIONS = {'.mp3', '.m4a', '.flac', '.wav', '.ogg', '.aac', '.wma', '.opus'}


class MusicLibraryScanner:
    """Scans and indexes local music files"""
    
    def __init__(self, index_file: str = "./data/music_index.json"):
        self.index_file = Path(index_file)
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.library: Dict[str, Dict] = {}
        self._load_index()
    
    def _load_index(self):
        """Load existing music index"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.library = json.load(f)
                logger.info(f"Loaded {len(self.library)} songs from index")
            except Exception as e:
                logger.error(f"Error loading music index: {e}")
                self.library = {}
    
    def _save_index(self):
        """Save music index to file"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.library, f, indent=2)
            logger.info(f"Saved {len(self.library)} songs to index")
        except Exception as e:
            logger.error(f"Error saving music index: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate unique hash for file (path + size + mtime)"""
        stat = os.stat(file_path)
        hash_input = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _extract_basic_metadata(self, file_path: Path) -> Dict:
        """Extract basic metadata from file"""
        stat = file_path.stat()
        
        # Basic metadata from filename and path
        return {
            "file_path": str(file_path),
            "filename": file_path.name,
            "title": file_path.stem,  # filename without extension
            "extension": file_path.suffix.lower(),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "folder": file_path.parent.name,
            # Try to extract artist from folder structure (common pattern: Artist/Album/Song)
            "artist": file_path.parent.parent.name if len(file_path.parts) > 2 else "Unknown Artist",
            "album": file_path.parent.name,
        }
    
    def scan_directory(self, directory: str, rescan: bool = False) -> int:
        """
        Scan directory for audio files
        
        Args:
            directory: Path to scan
            rescan: Force rescan of all files
            
        Returns:
            Number of new files found
        """
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return 0
        
        logger.info(f"Scanning directory: {directory}")
        new_files = 0
        
        # Find all audio files
        audio_files = []
        for ext in AUDIO_EXTENSIONS:
            audio_files.extend(directory.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        for file_path in audio_files:
            try:
                file_hash = self._get_file_hash(str(file_path))
                
                # Skip if already indexed (unless rescanning)
                if not rescan and file_hash in self.library:
                    continue
                
                # Extract metadata
                metadata = self._extract_basic_metadata(file_path)
                metadata["hash"] = file_hash
                metadata["indexed_at"] = datetime.now().isoformat()
                
                # Add to library
                self.library[file_hash] = metadata
                new_files += 1
                
                if new_files % 100 == 0:
                    logger.info(f"Indexed {new_files} new files...")
                    self._save_index()  # Periodic save
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Final save
        self._save_index()
        logger.info(f"Scan complete. Added {new_files} new files. Total: {len(self.library)}")
        
        return new_files
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Simple search for songs by title, artist, or album
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching songs
        """
        query = query.lower()
        results = []
        
        for song in self.library.values():
            # Search in title, artist, album, filename
            searchable = f"{song.get('title', '')} {song.get('artist', '')} {song.get('album', '')} {song.get('filename', '')}".lower()
            
            if query in searchable:
                results.append(song)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_random_songs(self, count: int = 10, genre: Optional[str] = None) -> List[Dict]:
        """Get random songs from library"""
        import random
        
        songs = list(self.library.values())
        
        # Filter by genre/folder if specified
        if genre:
            songs = [s for s in songs if genre.lower() in s.get('folder', '').lower()]
        
        # Shuffle and return requested count
        random.shuffle(songs)
        return songs[:count]
    
    def get_stats(self) -> Dict:
        """Get library statistics"""
        if not self.library:
            return {"total_songs": 0}
        
        total_size = sum(s.get('size_mb', 0) for s in self.library.values())
        artists = set(s.get('artist', 'Unknown') for s in self.library.values())
        albums = set(s.get('album', 'Unknown') for s in self.library.values())
        
        return {
            "total_songs": len(self.library),
            "total_size_gb": round(total_size / 1024, 2),
            "unique_artists": len(artists),
            "unique_albums": len(albums),
            "file_types": dict(self._count_extensions())
        }
    
    def _count_extensions(self) -> Dict[str, int]:
        """Count files by extension"""
        from collections import Counter
        extensions = [s.get('extension', '') for s in self.library.values()]
        return Counter(extensions)
    
    def watch_usb_drives(self) -> List[str]:
        """
        Detect USB drives (macOS specific)
        
        Returns:
            List of USB mount points
        """
        volumes = Path("/Volumes")
        usb_drives = []
        
        if volumes.exists():
            for mount in volumes.iterdir():
                if mount.is_dir() and not mount.name.startswith('.'):
                    # Check if it's likely a USB drive (not system volume)
                    if mount.name not in ["Macintosh HD", "Data"]:
                        usb_drives.append(str(mount))
        
        return usb_drives


# Example usage
if __name__ == "__main__":
    scanner = MusicLibraryScanner()
    
    # Scan user's Music folder
    music_folder = Path.home() / "Music"
    if music_folder.exists():
        scanner.scan_directory(str(music_folder))
    
    # Show stats
    stats = scanner.get_stats()
    print(f"Library stats: {json.dumps(stats, indent=2)}")
    
    # Search example
    results = scanner.search("jazz")
    print(f"\nFound {len(results)} jazz songs")