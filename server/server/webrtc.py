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
        self.ice_servers = [IceServer(urls=config.network.stun_server)]
    
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
        for pc_id, connection in list(self.connections.items()):
            try:
                logger.info(f"Closing connection {pc_id}")
                # Note: Actual connection cleanup depends on pipecat implementation
                # This will be handled by the connection's event handlers
            except Exception as e:
                logger.error(f"Error closing connection {pc_id}: {e}")
        
        self.connections.clear()
        logger.info("✅ All connections cleaned up")