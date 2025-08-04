"""
Music control tools for the AI DJ system
"""

from typing import Dict, Any, List, Optional
from loguru import logger
from music.library_scanner import MusicLibraryScanner

# Global instances (will be set by pipeline)
_audio_player = None
_music_scanner = None


def set_music_system(audio_player, scanner: MusicLibraryScanner):
    """Set the global music system instances"""
    global _audio_player, _music_scanner
    _audio_player = audio_player
    _music_scanner = scanner
    logger.info("Music system configured for tools")


async def play_music(query: Optional[str] = None) -> Dict[str, Any]:
    """
    Play music - either resume current or search and play new song
    
    Args:
        query: Optional search query (e.g., "jazz", "Beatles", "upbeat music")
        
    Returns:
        Playback status
    """
    if not _audio_player or not _music_scanner:
        return {"error": "Music system not configured"}
    
    try:
        # If no query, play a random song
        if not query:
            results = _music_scanner.get_random_songs(1)
            if not results:
                return {"error": "No music found in library"}
            song = results[0]
        else:
            # Search for specific music
            results = _music_scanner.search(query, limit=1)
            if not results:
                # Try random if no specific match
                results = _music_scanner.get_random_songs(1)
                if not results:
                    return {"error": "No music found in library"}
            song = results[0]
        
        # Play using the simple player for now
        from processors.music_player_simple import play_music_simple
        logger.info(f"🎵 Playing file: {song['file_path']}")
        status = play_music_simple(song["file_path"], song)
        
        return {
            "success": True,
            "action": "playing",
            "song": {
                "title": song.get("title", "Unknown"),
                "artist": song.get("artist", "Unknown Artist"),
                "album": song.get("album", "Unknown Album"),
                "file": song.get("filename", "Unknown")
            },
            "message": f"Now playing: {song.get('title', 'Unknown')} by {song.get('artist', 'Unknown Artist')}"
        }
        
    except Exception as e:
        logger.error(f"Error playing music: {e}")
        return {"error": str(e)}


async def pause_music() -> Dict[str, Any]:
    """Pause current playback"""
    try:
        from processors.music_player_simple import get_player
        player = get_player()
        player.pause()
        return {"success": True, "message": "Music paused"}
    except Exception as e:
        logger.error(f"Error pausing music: {e}")
        return {"error": str(e)}


async def skip_song() -> Dict[str, Any]:
    """Skip to next song in queue"""
    if not _audio_player:
        return {"error": "Music system not configured"}
    
    try:
        from processors.audio_player_real import MusicControlFrame
        await _audio_player.push_frame(MusicControlFrame("skip"))
        return {"success": True, "message": "Skipping to next song"}
    except Exception as e:
        logger.error(f"Error skipping song: {e}")
        return {"error": str(e)}


async def queue_music(query: str) -> Dict[str, Any]:
    """
    Add songs to the play queue
    
    Args:
        query: Search query for songs to queue
        
    Returns:
        Queue status
    """
    if not _audio_player or not _music_scanner:
        return {"error": "Music system not configured"}
    
    try:
        # Search for music
        results = _music_scanner.search(query, limit=5)
        if not results:
            return {"error": f"No music found matching '{query}'"}
        
        # Queue all results
        queued = []
        from processors.audio_player_real import MusicControlFrame
        
        for song in results:
            await _audio_player.push_frame(MusicControlFrame("queue", {"file_path": song["file_path"], **song}))
            queued.append(song.get("title", "Unknown"))
        
        return {
            "success": True,
            "queued_count": len(queued),
            "songs": queued,
            "message": f"Added {len(queued)} songs to queue"
        }
        
    except Exception as e:
        logger.error(f"Error queuing music: {e}")
        return {"error": str(e)}


async def search_music(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search music library
    
    Args:
        query: Search query
        limit: Maximum results
        
    Returns:
        Search results
    """
    if not _music_scanner:
        return {"error": "Music system not configured"}
    
    try:
        results = _music_scanner.search(query, limit=limit)
        
        return {
            "success": True,
            "count": len(results),
            "results": [
                {
                    "title": song.get("title", "Unknown"),
                    "artist": song.get("artist", "Unknown Artist"),
                    "album": song.get("album", "Unknown Album"),
                    "duration": song.get("duration", "Unknown")
                }
                for song in results
            ]
        }
        
    except Exception as e:
        logger.error(f"Error searching music: {e}")
        return {"error": str(e)}


async def get_now_playing() -> Dict[str, Any]:
    """Get current playing song and queue info"""
    try:
        from processors.music_player_simple import get_player
        player = get_player()
        status = player.get_status()
        current = status.get("current_song")
        
        if not current or not status.get("is_playing"):
            return {
                "success": True,
                "is_playing": False,
                "message": "Nothing is currently playing"
            }
        
        return {
            "success": True,
            "is_playing": status.get("is_playing", False),
            "is_paused": status.get("is_paused", False),
            "current_song": {
                "title": current.get("title", "Unknown"),
                "artist": current.get("artist", "Unknown Artist"),
                "album": current.get("album", "Unknown Album")
            },
            "volume": status.get("volume", 70)
        }
        
    except Exception as e:
        logger.error(f"Error getting now playing: {e}")
        return {"error": str(e)}


async def set_volume(level: int) -> Dict[str, Any]:
    """
    Set playback volume
    
    Args:
        level: Volume level (0-100)
        
    Returns:
        Volume status
    """
    try:
        # Convert percentage to 0-1 range
        volume = max(0, min(100, level)) / 100.0
        
        from processors.music_player_simple import get_player
        player = get_player()
        player.set_volume(volume)
        
        return {
            "success": True,
            "volume": level,
            "message": f"Volume set to {level}%"
        }
        
    except Exception as e:
        logger.error(f"Error setting volume: {e}")
        return {"error": str(e)}


async def create_playlist(mood: str, count: int = 10) -> Dict[str, Any]:
    """
    Create a playlist based on mood/genre
    
    Args:
        mood: Mood or genre (e.g., "relaxing", "energetic", "jazz")
        count: Number of songs
        
    Returns:
        Created playlist
    """
    if not _music_scanner or not _audio_player:
        return {"error": "Music system not configured"}
    
    try:
        # Search by mood/genre
        songs = _music_scanner.search(mood, limit=count)
        
        # If not enough specific matches, add random songs
        if len(songs) < count:
            additional = _music_scanner.get_random_songs(count - len(songs))
            songs.extend(additional)
        
        # Queue all songs
        from processors.audio_player_real import MusicControlFrame
        for song in songs:
            await _audio_player.push_frame(MusicControlFrame("queue", {"file_path": song["file_path"], **song}))
        
        return {
            "success": True,
            "playlist_name": f"{mood.title()} Vibes",
            "song_count": len(songs),
            "songs": [s.get("title", "Unknown") for s in songs[:5]],  # First 5
            "message": f"Created {mood} playlist with {len(songs)} songs"
        }
        
    except Exception as e:
        logger.error(f"Error creating playlist: {e}")
        return {"error": str(e)}


async def get_music_stats() -> Dict[str, Any]:
    """Get music library statistics"""
    if not _music_scanner:
        return {"error": "Music system not configured"}
    
    try:
        stats = _music_scanner.get_stats()
        
        return {
            "success": True,
            "library_stats": {
                "total_songs": stats.get("total_songs", 0),
                "total_size_gb": stats.get("total_size_gb", 0),
                "unique_artists": stats.get("unique_artists", 0),
                "unique_albums": stats.get("unique_albums", 0)
            },
            "message": f"Your library has {stats.get('total_songs', 0)} songs from {stats.get('unique_artists', 0)} artists"
        }
        
    except Exception as e:
        logger.error(f"Error getting music stats: {e}")
        return {"error": str(e)}