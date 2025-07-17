#!/usr/bin/env python3
import uvicorn
from src.api.server import app
from src.utils.logging import setup_logging
from src.config.settings import Settings

if __name__ == "__main__":
    setup_logging()
    settings = Settings()
    
    is_valid, error_msg = settings.validate()
    if not is_valid:
        print(f"Configuration error: {error_msg}")
        exit(1)
    
    print(f"Starting Swarm of Experts API server on {settings.server_host}:{settings.server_port}")
    print(f"Available models: swarm-lite, groq-swarm")
    
    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.server_port,
        log_level="info",
        reload=False,
        workers=settings.server_workers
    )