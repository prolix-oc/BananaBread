import signal
import threading
import uvicorn
import logging
import atexit

from bananabread.config import logger
from bananabread.api import cleanup_resources, shutdown_event

# ----- Custom Uvicorn Server \u2014

class CustomServer(uvicorn.Server):
    """Custom uvicorn server to control startup logging timing and graceful shutdown"""
    
    def __init__(self, config):
        super().__init__(config)
        self._shutdown_requested = False
        self._shutdown_lock = threading.Lock()
    
    async def startup(self, sockets=None):
        """Override startup to add our custom logging at the right time"""
        # Call the original startup
        await super().startup(sockets)
        
        # Now that the server is actually starting up, log our messages
        logger.info("🍞 BananaBread-Emb server is now running!")
        logger.info(f"🌐 Server available at: http://{self.config.host}:{self.config.port}")
        logger.info(f"📚 API Documentation: http://{self.config.host}:{self.config.port}/docs")
        logger.info(f"📖 ReDoc Documentation: http://{self.config.host}:{self.config.port}/redoc")
    
    async def shutdown(self, sockets=None):
        """Override shutdown to ensure graceful cleanup"""
        with self._shutdown_lock:
            if self._shutdown_requested:
                logger.debug("Shutdown already in progress, skipping duplicate shutdown")
                return
            
            self._shutdown_requested = True
            logger.info("🛑 Server shutdown initiated...")
        
        # Call the original shutdown first to stop accepting new requests
        try:
            await super().shutdown(sockets)
        except Exception as e:
            logger.error(f"Error during uvicorn shutdown: {e}")
        
        # Then call our custom cleanup
        cleanup_resources()
        
        logger.info("✅ Server shutdown completed")
    
    def handle_exit(self, sig, frame):
        """Handle exit signals more gracefully"""
        signal_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else str(sig)
        logger.info(f"📡 Received exit signal {signal_name}, shutting down server...")
        self.should_exit = True

def main():
    """Main entry point for the bananabread-emb console script"""
    global server
    
    # Create uvicorn configuration
    # Note: We use the import string "bananabread.api:app" now
    config = uvicorn.Config(
        "bananabread.api:app",
        host="0.0.0.0",
        port=8008,
        reload=False,
        log_config=None  # Disable uvicorn's default logging
    )
    
    # Create custom server
    server = CustomServer(config)
    
    # Set up signal handling for the server
    def server_signal_handler(signum, frame):
        """Handle signals for server shutdown - minimal work here"""
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.info(f"📡 Received {signal_name} signal, initiating server shutdown...")
        
        # Only set the shutdown flag - don't call cleanup directly
        # Cleanup will be handled by uvicorn's shutdown sequence
        server.should_exit = True
        # Also update the global shutdown event
        shutdown_event.set()
    
    # Override the signal handlers to use our server-specific handler
    signal.signal(signal.SIGINT, server_signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, server_signal_handler)  # Termination signal
    
    # Also handle SIGBREAK on Windows
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, server_signal_handler)
    
    # Safety definitions for signal handler references in cleanup
    def signal_handler_cleanup(signum, frame):
        server_signal_handler(signum, frame)

    # Register cleanup function for normal exit (fallback)
    atexit.register(cleanup_resources)
    
    logger.info("🚀 Initializing BananaBread-Emb server...")
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("⌨️  Received KeyboardInterrupt, shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Unexpected error during server execution: {e}")
        raise
    finally:
        # Ensure cleanup always runs at the end
        logger.info("🧹 Running final cleanup...")
        cleanup_resources()

if __name__ == "__main__":
    main()
