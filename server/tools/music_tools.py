"""
Music control tools for the AI DJ system
"""

from typing import Dict, Any, List, Optional
from loguru import logger
from music.library_scanner import MusicLibraryScanner
from processors.music_player_simple import get_player

# Global instances (will be set by pipeline)
_music_scanner: Optional[MusicLibraryScanner] = None


def set_music_scanner(scanner: MusicLibraryScanner):
    """Set the global music scanner instance."""
    global _music_scanner
    _music_scanner = scanner
    logger.info("Music scanner configured for tools")


async def play_music(query: Optional[str] = None) -> Dict[str, Any]:
    """
    Play music - either resume current or search and play new song
    
    Args:
        query: Optional search query (e.g., "jazz", "Beatles", "upbeat music")
        
    Returns:
        Playback status
    """
    player = get_player()
    if not _music_scanner:
        return {"error": "Music system not configured"}
    
    try:
        # If no query, resume or play a random song
        if not query:
            status = player.get_status()
            if status.get("is_playing") and status.get("is_paused"):
                player.resume()
                return {"success": True, "message": "Resuming playback."}
            
            # Get current queue info to avoid recent songs
            current_song = status.get("current_song")
            
            # Build set of recent file paths to avoid
            recent_paths = set()
            if current_song and current_song.get("file_path"):
                recent_paths.add(current_song["file_path"])
            
            # Get multiple random songs and pick one that's not recent
            results = _music_scanner.get_random_songs(10)
            if not results:
                return {"error": "No music found in library"}
            
            song = next((s for s in results if s.get("file_path") not in recent_paths), results[0])

        else:
            # Search for specific music
            results = _music_scanner.search(query, limit=1)
            if not results:
                return {"error": f"No music found for '{query}'"}
            song = results[0]
        
        logger.info(f"🎵 Playing file: {song['file_path']}")
        player.play(song["file_path"], song)
        
        return {
            "success": True,
            "action": "playing",
            "song": {
                "title": song.get("title", "Unknown"),
                "artist": song.get("artist", "Unknown Artist"),
            },
            "message": f"Now playing: {song.get('title', 'Unknown')}"
        }
        
    except Exception as e:
        logger.error(f"Error playing music: {e}")
        return {"error": str(e)}


async def pause_music() -> Dict[str, Any]:
    """Pause current playback"""
    try:
        get_player().pause()
        return {"success": True, "message": "Music paused"}
    except Exception as e:
        logger.error(f"Error pausing music: {e}")
        return {"error": str(e)}


async def skip_song() -> Dict[str, Any]:
    """Skip to next song in queue"""
    logger.info("⏭️⏭️⏭️ SKIP_SONG CALLED")
    try:
        get_player().skip_to_next()
        return {"success": True, "message": "Skipping to next song"}
    except Exception as e:
        logger.error(f"Error skipping song: {e}")
        return {"error": str(e)}


async def stop_music() -> Dict[str, Any]:
    """Stop music playback completely"""
    logger.info("🛑🛑🛑 STOP_MUSIC CALLED")
    try:
        get_player().stop()
        return {"success": True, "message": "Music stopped"}
    except Exception as e:
        logger.error(f"Error stopping music: {e}")
        return {"error": str(e)}


async def queue_music(query: str) -> Dict[str, Any]:
    """
    Add songs to the play queue (avoiding duplicates)
    
    Args:
        query: Search query for songs to queue
        
    Returns:
        Queue status
    """
    player = get_player()
    if not _music_scanner:
        return {"error": "Music system not configured"}
    
    try:
        # Get current queue info to check for duplicates
        status = player.get_status()
        current_song = status.get("current_song")
        current_queue = status.get("queue", [])
        
        # Build set of already queued file paths
        already_queued = set()
        if current_song and current_song.get("file_path"):
            already_queued.add(current_song["file_path"])
        for song in current_queue:
            if song.get("file_path"):
                already_queued.add(song["file_path"])
        
        # Search for music
        results = _music_scanner.search(query, limit=10)
        if not results:
            return {"error": f"No music found matching '{query}'"}
        
        # Filter out duplicates and queue unique songs
        queued_count = 0
        for song in results:
            if song.get("file_path") not in already_queued:
                player.queue_song(song["file_path"], song)
                already_queued.add(song["file_path"])
                queued_count += 1
                if queued_count >= 5:
                    break
        
        if queued_count == 0:
            return {"success": False, "message": "All matching songs are already in the queue."}
        
        return {"success": True, "message": f"Added {queued_count} new songs to the queue."}
        
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
    logger.info(f"🎵🎵🎵 CREATE_PLAYLIST CALLED with mood='{mood}', count={count}")
    
    player = get_player()
    if not _music_scanner:
        return {"error": "Music system not configured"}
    
    try:
        # Get current queue info to check for duplicates
        status = player.get_status()
        current_song = status.get("current_song")
        current_queue = status.get("queue", [])
        
        # Build set of already queued file paths
        already_queued = set()
        if current_song and current_song.get("file_path"):
            already_queued.add(current_song["file_path"])
        for song in current_queue:
            if song.get("file_path"):
                already_queued.add(song["file_path"])
        
        # Search by mood/genre - get extra to account for duplicates
        songs = _music_scanner.search(mood, limit=count * 2)
        
        # If not enough specific matches, add random songs
        if len(songs) < count * 2:
            additional = _music_scanner.get_random_songs(count * 2 - len(songs))
            songs.extend(additional)
        
        # Queue unique songs
        queued = []
        queued_titles = []
        
        for song in songs:
            file_path = song.get("file_path")
            if file_path and file_path not in already_queued:
                player.queue_song(file_path, song)
                queued.append(song)
                queued_titles.append(song.get("title", "Unknown"))
                already_queued.add(file_path)
                
                if len(queued) >= count:
                    break
        
        # If nothing is playing, start the first song
        if not status.get("is_playing") and queued:
            logger.info("🎵 Starting playback from new playlist")
            player.skip_to_next() # This will start the first song in the queue
        
        return {
            "success": True,
            "playlist_name": f"{mood.title()} Vibes",
            "song_count": len(queued),
            "songs": queued_titles[:5],  # First 5
            "message": f"Created {mood} playlist with {len(queued)} songs"
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