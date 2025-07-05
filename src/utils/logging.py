import logging
import os
import sys
from pathlib import Path

def setup_logging(level=logging.INFO):
    os.environ.setdefault("GRPC_VERBOSITY", "NONE")
    os.environ.setdefault("GRPC_TRACE", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    handlers = [logging.FileHandler(log_dir / "swarm.log")]
    if debug_mode:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)