"""
WebRTC connection management
"""

from typing import Dict
from loguru import logger
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection, IceServer
from config import config


class WebRTCManager:
    """Manages WebRTC connections and peer connection lifecycle"""
    
    def __init__(self):
        self.connections: Dict[str, SmallWebRTCConnection] = {}
        # Support multiple ICE servers and optional TURN from config/env
        ice_urls = []
        if hasattr(config.network, "stun_servers") and config.network.stun_servers:
            ice_urls.extend(config.network.stun_servers)
        elif hasattr(config.network, "stun_server"):
            ice_urls.append(config.network.stun_server)
        turn_url = getattr(config.network, "turn_server", None)
        if turn_url:
            ice_urls.append(turn_url)
        self.ice_servers = [IceServer(urls=url) for url in ice_urls]
    
    async def handle_offer(self, request: dict) -> dict:
        """
        Handle WebRTC offer and manage connection lifecycle
        
        Args:
            request: WebRTC offer request containing SDP and optional pc_id
            
        Returns:
            WebRTC answer with connection details
        """
        pc_id = request.get("pc_id")
        
        if pc_id and pc_id in self.connections:
            # Reuse existing connection
            connection = self.connections[pc_id]
            logger.info(f"Reusing existing connection for pc_id: {pc_id}")
            await connection.renegotiate(
                sdp=request["sdp"], 
                type=request["type"], 
                restart_pc=request.get("restart_pc", False)
            )
        else:
            # Create new connection
            connection = SmallWebRTCConnection(self.ice_servers)
            await connection.initialize(sdp=request["sdp"], type=request["type"])
            
            # Setup cleanup handler
            @connection.event_handler("closed")
            async def handle_disconnected(conn: SmallWebRTCConnection):
                logger.info(f"Discarding peer connection for pc_id: {conn.pc_id}")
                self.connections.pop(conn.pc_id, None)
        
        # Get answer and store connection
        answer = connection.get_answer()
        self.connections[answer["pc_id"]] = connection
        
        return answer, connection
    
    def get_connection(self, pc_id: str) -> SmallWebRTCConnection:
        """Get connection by ID"""
        return self.connections.get(pc_id)
    
    def remove_connection(self, pc_id: str) -> bool:
        """Remove connection by ID"""
        if pc_id in self.connections:
            del self.connections[pc_id]
            return True
        return False
    
    def get_active_connections(self) -> Dict[str, SmallWebRTCConnection]:
        """Get all active connections"""
        return self.connections.copy()
    
    async def cleanup_all_connections(self):
        """Cleanup all active connections"""
        logger.info(f"Cleaning up {len(self.connections)} active connections...")
        import inspect
        for pc_id, connection in list(self.connections.items()):
            try:
                logger.info(f"Closing connection {pc_id}")
                if hasattr(connection, "close"):
                    res = connection.close()
                    if inspect.isawaitable(res):
                        await res
            except Exception as e:
                logger.error(f"Error closing connection {pc_id}: {e}")
            finally:
                self.connections.pop(pc_id, None)
        
        logger.info("✅ All connections cleaned up")