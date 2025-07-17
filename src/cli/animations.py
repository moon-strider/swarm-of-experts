import sys
import threading
import time
from typing import List, Optional
from abc import ABC, abstractmethod

from .ui import Color


class Animation(ABC):
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
        
    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        self._clear_line()
        
    @abstractmethod
    def _animate(self) -> None:
        pass
        
    def _clear_line(self) -> None:
        sys.stdout.write('\r\033[K')
        sys.stdout.write('\r')
        sys.stdout.flush()
        sys.stderr.flush()


class SpinnerAnimation(Animation):
    def __init__(self, text: str = "Thinking", frames: Optional[List[str]] = None):
        super().__init__()
        self.text = text
        self.frames = frames or ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        
    def _animate(self) -> None:
        time.sleep(0.02)
        frame_index = 0
        while self.running:
            frame = self.frames[frame_index % len(self.frames)]
            sys.stdout.write(f'\r\033[K{Color.CYAN.value}{frame} {self.text}...{Color.RESET.value}')
            sys.stdout.flush()
            time.sleep(0.1)
            frame_index += 1